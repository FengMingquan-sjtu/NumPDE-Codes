import copy

import sympy as sym
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import meshpy.triangle as triangle
from scipy.interpolate import LinearNDInterpolator

# --------- Mesh Generator --------- 

def make_mesh(points):
    """ Generate triangle mesh.
    Input: 
        points: list of 4 tuples, the coordinates of 4 corners of domain. 
    Output:
        nodes: np array, shape=(n_nodes, 2), each row is the coordinates of node.
        elements: np array, shape=(n_elements, 3), each row is the indices(in the nodes array) of 3 nodes of the element.
    """
    points = copy.deepcopy(points)

    def round_trip_connect(start, end):
        return [(i, i+1) for i in range(start, end)] + [(end, start)]

    
    facets = round_trip_connect(0, len(points)-1)

    circ_start = len(points)
    points.extend(
            (0.25 * np.cos(angle), 0.25 * np.sin(angle))
            for angle in np.linspace(0, 2*np.pi, 30, endpoint=False))

    

    facets.extend(round_trip_connect(circ_start, len(points)-1))

    def needs_refinement(vertices, area):
        bary = np.sum(np.array(vertices), axis=0)/3
        max_area = 0.01 + la.norm(bary, np.inf)*0.01
        return bool(area > max_area)

    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(facets)

    built_mesh = triangle.build(info, refinement_func=needs_refinement)
    nodes, elements = np.array(built_mesh.points), np.array(built_mesh.elements)
    return nodes, elements


# --------- Sparse Matrix --------- 
class MatrixBuilder:  #sparse matrix helper 
    def __init__(self):
        self.rows = []
        self.cols = []
        self.vals = []
        
    def add(self, rows, cols, submat):
        """ add a local matrix.
        input: 
            rows, cols: list of global node indices.
            submat: local matrix 
        """
        for i, ri in enumerate(rows):
            for j, cj in enumerate(cols):
                self.rows.append(ri)
                self.cols.append(cj)
                self.vals.append(submat[i, j])
                
    def coo_matrix(self):
        return sparse.coo_matrix((self.vals, (self.rows, self.cols)))


# --------- FEM Solver V1 --------- 
class FEMSolerV1:
    ''' FEM solver for poission equation:
            \grad kappa(x,y) \grad u = f.
                                u_0  = g.
    '''
    def __init__(self):
        #gradient of 1-order-poly basis functions.
        self.dbasis = np.array([
            [-1, 1, 0],  # dphi/dr
            [-1, 0, 1]]) # dphi/ds
    def setDomain(self,bound_points, bc_points):
        ''' set the domain, determine mesh and boundary. Only allow rectangular domain.
        Input:
            bound_points: a list of coordinates of the domain's corner. 
                E.g. [(-1, -1), (1, -1), (1, 1), (-1, 1)] denotes a rectangle centered at origin, side length=2.
            bc_points: a list of coordinates, along which line the boundary condition g(x,y) is satisfied.
                E.g. [(-9999, -1)] denotes that g(x,y) is satisfied at line y=-1, and x=-9999 is a dummy notation.
        Output:None 
        '''
        bound = np.array(bound_points)
        self.x_lb, self.x_ub, self.y_lb, self.y_ub = bound[:,0].min(), bound[:,0].max(), bound[:,1].min(), bound[:,1].max()
        self.nodes, self.elements = make_mesh(bound_points)
        X, Y = self.nodes[:,0], self.nodes[:,1]
        tol = 1e-12
        self.is_boundary, self.is_g_boundary = np.zeros(len(self.nodes), dtype=bool), np.zeros(len(self.nodes), dtype=bool)
        for x,y in bound_points:
            self.is_boundary |= (np.abs(X-x) < tol) | (np.abs(Y-y) < tol)

        for x,y in bc_points:
            self.is_g_boundary |= (np.abs(X-x) < tol) | (np.abs(Y-y) < tol)


    def solve(self, kappa, f, g, visualize=True, u_exact=None):
        ''' solve the given poisson equation
        Input:
            kappa,f,g: functions, whose input is (x,y) denoted by a (2, n) array, and output is (n,) array.
            visualize: boolean flag for plt 3d result.
            u_exact: exact solution function for ploting.
        Output:
            u: the nodal solution, (n_nodes,) array.
            nodes: the coordinates of nodes, (n_nodes, 2) array.
        '''
        # -------- Assemble the A matrix (LHS)  --------  
        a_builder = MatrixBuilder()

        for ei in range(0, len(self.elements)):
            vert_indices = self.elements[ei, :]
            x0, x1, x2 = el_verts = self.nodes[vert_indices]
            centroid = np.mean(el_verts, axis=0)

            J = np.array([x1-x0, x2-x0]).T
            invJT = la.inv(J.T)
            detJ = la.det(J)
            dphi = invJT @ self.dbasis

            Aelem = kappa(centroid) * (detJ / 2.0) * dphi.T @ dphi

            a_builder.add(vert_indices, vert_indices, Aelem)
        A = a_builder.coo_matrix().tocsr().tocoo() #Duplicate entries will be summed

        # -------- Assemble the b vector (RHS)  --------  
        b = np.zeros(len(self.nodes))

        for ei in range(0, len(self.elements)):
            vert_indices = self.elements[ei, :]
            x0, x1, x2 = el_verts = self.nodes[vert_indices]
            centroid = np.mean(el_verts, axis=0)

            J = np.array([x1-x0, x2-x0]).T
            detJ = la.det(J)

            belem = f(centroid) * (detJ / 6.0) * np.ones((3,))

            for i, vi in enumerate(vert_indices):
                b[vi] += belem[i]

        # -------- Encode Boundary Conditions  --------  
        u0 = np.zeros(len(self.nodes))
        u0[self.is_g_boundary] = g(self.nodes[self.is_g_boundary].T)

        rhs = b - A @ u0  #subtract additional term induced by u_0

        # default zero boundary
        rhs[self.is_boundary] = 0.0  #set b_i to zero
        for k in range(A.nnz): #nnz= number of non-zero elements.
            i = A.row[k]
            j = A.col[k]
            if self.is_boundary[i]:
                A.data[k] = 1 if i == j else 0  #set A[i,:] to one-hot $\mathbb{1}_i$.

        # -------- Solve Sparse System  --------  
        uhat = sla.spsolve(A.tocsr(), rhs)
        u = uhat + u0  #reconstruct solution u from \hat u and u_0

        # -------- Evaluate & Visualize  -------- 
        # test grid points
        X = np.linspace(self.x_lb, self.x_ub, 100)
        Y = np.linspace(self.y_lb, self.y_ub, 100)
        X, Y = np.meshgrid(X, Y)

        # MAE on grid points
        if not u_exact is None:
            u_interp = LinearNDInterpolator(self.nodes, u)(X,Y)
            err = u_interp -  u_exact((X,Y))
            err = np.abs(err).mean()
            print("Mean Abs Error=", err)

        # visualization
        if visualize:
            fig = plt.figure(figsize=(8,8))
            ax1 = fig.add_subplot(projection='3d')
            ax1.plot_trisurf(self.nodes[:,0], self.nodes[:,1], u, triangles=self.elements, cmap=plt.cm.jet, linewidth=0.2)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('U')
            ax1.set_title("FEM solution")
            if not u_exact is None:
                fig = plt.figure(figsize=(8,8))
                ax2 = fig.add_subplot(projection='3d')
                
                ax2.plot_surface(X, Y, u_exact((X,Y)), cmap=plt.cm.jet, linewidth=0.1)
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.set_zlabel('U')
                ax2.set_title("Exact solution")
            plt.show()
        
        
        return u, self.nodes


def test_V1_case1():
    def kappa(xvec):  #left-hand-side(lhs) non-linear func 
        x, y = xvec
        if (x**2 + y**2)**0.5 <= 0.25:
            return 25.0
        else:
            return 1.0

    def f(xvec):    #rhs func
        x, y = xvec
        if (x**2 + y**2)**0.5 <= 0.25:
            return 100.0
        else:
            return 0.0

        
    def g(xvec):  #boundary condition
        x, y = xvec
        return 1 * (1 - x**2) 

    s = FEMSolerV1()
    s.setDomain([(-1, -1), (1, -1), (1, 1), (-1, 1)], [(-9999, -1)])
    s.solve(kappa, f, g)



def test_V1_case2():
    x, y = sym.symbols('x y')
    u = 1 + sym.sin(2*x) + sym.sin(y)
    kappa = sym.sin(x) + sym.cos(y)
    f = - sym.diff(kappa*sym.diff(u, x), x) - sym.diff(kappa*sym.diff(u, y), y)
    f = sym.simplify(f)
    g = u
    print('kappa=',kappa)
    print("f=",f)
    print("u_exact=", u)
    kappa = sym.lambdify([(x, y)], kappa, 'numpy')
    f = sym.lambdify([(x, y)], f, 'numpy')
    g = sym.lambdify([(x, y)], g, 'numpy')
    u = sym.lambdify([(x, y)], u, 'numpy')

    s = FEMSolerV1()
    s.setDomain([(-1, -1), (1, -1), (1, 1), (-1, 1)], [(-1, -1), (1, -1), (1, 1), (-1, 1)])
    s.solve(kappa, f, g, u_exact=u)


if __name__ == "__main__":
    test_V1_case2()