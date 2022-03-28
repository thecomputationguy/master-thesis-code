import numpy as np
from scipy import sparse


def adjacency_hypercube_lattice(length, ndim):
    """
    Construct the adjacency matrix for a hypercube lattice with periodic boundary conditions.
    """
    L = length**ndim
    adj = np.zeros((L, L), dtype=int)
    idx = np.arange(L).reshape(ndim*[length])
    for d in range(ndim):
        for s in [-1, 1]:
            ids = np.roll(idx, s, axis=d)
            for (i, j) in zip(idx.reshape(-1), ids.reshape(-1)):
                adj[i, j] = 1
    return adj


def construct_ising_hamiltonian(J, g, adj):
    """
    Construct Ising Hamiltonian as sparse matrix.
    """
    # Pauli-X and Z matrices
    X = sparse.csr_matrix([[0,  1], [1,  0]], dtype=float)
    Z = sparse.csr_matrix([[1,  0], [0, -1]], dtype=float)
    # number of lattice sites
    L = adj.shape[0]
    H = sparse.csr_matrix((2**L, 2**L), dtype=float)
    for i in range(L):
        for j in range(i+1, L):
            if adj[i, j] == 1:
                H -= J * sparse.kron(sparse.eye(2**i),
                         sparse.kron(Z,
                         sparse.kron(sparse.eye(2**(j-i-1)),
                         sparse.kron(Z,
                                     sparse.eye(2**(L-j-1))))))
    # external field
    for i in range(L):
        H -= g * sparse.kron(sparse.eye(2**i), sparse.kron(X, sparse.eye(2**(L-i-1))))
    return H


def construct_ising_local_term(J, g):
    """
    Construct local interaction term of Ising Hamiltonian on a one-dimensional
    lattice for interaction parameter `J` and external field parameter `g`.
    """
    # Pauli-X and Z matrices
    X = np.array([[0.,  1.], [1.,  0.]])
    Z = np.array([[1.,  0.], [0., -1.]])
    I = np.identity(2)
    return -J*np.kron(Z, Z) - g*0.5*(np.kron(X, I) + np.kron(I, X))
