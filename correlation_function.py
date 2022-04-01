import numpy as np
import scipy as sp
import scipy.sparse.linalg as sparse
from hamiltonian import adjacency_hypercube_lattice, construct_ising_hamiltonian

def time_correlation(lattice_size : int, step_size : float, num_steps : int, psi0 : np.ndarray, site_operator : np.ndarray, hamiltonian : sp.sparse.csr_matrix) -> np.ndarray:
    """Calculates time-ordered dynamical correlations.

    Parameters
    ----------
    lattice_size : int
        Number of sites in the lattice.
    step_size : float
        step-size for simulation.
    num_steps : int
        number of simulation steps.
    psi0 : np.ndarray
        initial state of the system.
    site_operator : np.ndarray
        perturbation to be applied.
    hamiltonian : sp.sparse.csr_matrix
        hamiltonian of the system.

    Returns
    -------
    np.ndarray
        correlation function for all points in the lattice, for all time-steps.
    """    

    # Normalize the state vector
    psi = psi0 / np.linalg.norm(psi0) 

    # Data Structures to store correlation values
    correlation_data = np.zeros((num_steps, lattice_size), dtype = "complex_")

    # The perturbation is applied to the middle of the lattice
    j = lattice_size / 2 if lattice_size % 2 == 0 else (lattice_size-1) / 2

    # Initial perturbation operator and the initial intermediate state.
    sigma_j = sp.sparse.kron(sp.sparse.eye(2**j), sp.sparse.kron(site_operator, sp.sparse.eye(2**(lattice_size-j-1))))
    intermediate_state = sigma_j @ psi

    print(f'\nTime Evolution : Started')
    for step in range(num_steps):
        for i in range(lattice_size):
            # Local operator for the site
            sigma_i = sp.sparse.kron(sp.sparse.eye(2**i), sp.sparse.kron(site_operator, sp.sparse.eye(2**(lattice_size-i-1))))

            correlation_data[step, i] = np.vdot(psi, sigma_i @ intermediate_state)

        # Time Evolve the quantum state
        psi = sparse.expm_multiply(-1j * hamiltonian * step_size, psi)
        intermediate_state = sparse.expm_multiply(-1j * hamiltonian * step_size, intermediate_state)
    print('\nTime Evolution : Finished')

    return correlation_data