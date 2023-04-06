import copy

import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import meshpy.triangle as triangle


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
    def __init__(self):
        self.dbasis = np.array([
            [-1, 1, 0],  # dphi/dr
            [-1, 0, 1]]) # dphi/ds
    def setDomain(self,bound_points, bc_points):
        '''

        bound_points = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        bc_points = [(-9999, -1)]
        '''
        self.nodes, self.elements = make_mesh(bound_points)
        X, Y = self.nodes[:,0], self.nodes[:,1]
        tol = 1e-12
        self.is_boundary, self.is_g_boundary = np.zeros(len(self.nodes), dtype=bool), np.zeros(len(self.nodes), dtype=bool)
        for x,y in bound_points:
            self.is_boundary |= (np.abs(X-x) < tol) | (np.abs(Y-y) < tol)

        for x,y in bc_points:
            self.is_g_boundary |= (np.abs(X-x) < tol) | (np.abs(Y-y) < tol)


    def solve(self, kappa, f, g, visualize=True):
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
        A = a_builder.coo_matrix().tocsr().tocoo() #Duplicate entries will be summed together

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
        rhs[self.is_boundary] = 0.0  #set b_i to zero

        for k in range(A.nnz): #nnz= number of non-zero elements.
            i = A.row[k]
            j = A.col[k]
            if self.is_boundary[i]:
                A.data[k] = 1 if i == j else 0  #set A[i,:] to one-hot $\mathbb{1}_i$.

        # -------- Solve Sparse System  --------  
        uhat = sla.spsolve(A.tocsr(), rhs)
        u = uhat + u0  #reconstruct solution u from \hat u and u_0

        # -------- Visualize  -------- 
        if visualize:
            fig = plt.figure(figsize=(8,8))
            ax = plt.gca(projection='3d')
            ax.plot_trisurf(self.nodes[:,0], self.nodes[:,1], u, triangles=self.elements, cmap=plt.cm.jet, linewidth=0.2)
            plt.show()


if __name__ == "__main__":
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
