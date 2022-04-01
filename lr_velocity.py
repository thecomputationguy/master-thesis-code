import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_light_cone(correlations : np.ndarray, stepsize : float, num_steps : int, lattice_size : int, h : float, J : float, threshold : float = 0.009):
    """Plot the light cones from the correlation data.

    Parameters
    ----------
    correlations : np.ndarray
        correlation data.
    stepsize : float
        step-size of each simulation step.
    num_steps : inte
        number of simulation steps.
    lattice_size : int
        Number of sites in the lattice. 
    h : float
        interaction strength with the external field.
    J : float
        coupling strength with the nearest neighbor.
    threshold : float, optional
        threshold for the correlation, by default 0.009

    Returns
    -------
    tuple
        LR velocity : theoretical and extimated.
    """     
    v , v_paper = calculate_lr_velocity(correlations, stepsize, lattice_size, h, J, threshold)

    fig, axs = plt.subplots(1,2, figsize=(20,8))
    fig.suptitle(f"Light Cone for h={h}, J={J}", fontsize=18)
    extent = [-lattice_size, lattice_size, stepsize * num_steps, 0]
    cmap = mpl.cm.get_cmap('CMRmap')
    normalizer = mpl.colors.Normalize(0,1)
    cbar = mpl.cm.ScalarMappable(norm=normalizer, cmap=cmap)
    
    img_real = axs[0].imshow(np.real(correlations), extent=extent, aspect="auto", cmap=cmap, norm=normalizer)
    axs[0].set_xlabel('distance (i-j)', fontsize=20)
    axs[0].set_ylabel('Time (s)', fontsize=20)
    axs[0].set_title("Real part", fontsize=20)
    axs[0].axline((0,0), slope=1/v, color='r', label='Simulation')
    axs[0].axline((0,0), slope=-1/v, color='r')
    axs[0].axline((0,0), slope=1/v_paper, color='g', label='Wang et al.')
    axs[0].axline((0,0), slope=-1/v_paper, color='g')
    axs[0].legend(fontsize=16)
    
    img_abs = axs[1].imshow(np.abs(correlations), extent=extent, aspect="auto", cmap=cmap, norm=normalizer)
    axs[1].set_xlabel('distance (i-j)', fontsize=20)
    axs[1].set_ylabel('Time (s)')
    axs[1].set_title("Absolute Magnitude", fontsize=20)
    axs[1].axline((0,0), slope=1/v, color='r', label='Simulation')
    axs[1].axline((0,0), slope=-1/v, color='r')
    axs[1].axline((0,0), slope=1/v_paper, color='g', label='Wang et al.')
    axs[1].axline((0,0), slope=-1/v_paper, color='g')
    axs[1].legend(fontsize=16)
    
    fig.colorbar(cbar, ax=axs.ravel().tolist())
    plt.show()

    return v, v_paper


def calculate_lr_velocity(correlations : np.ndarray, stepsize : float, lattice_size : int, h : float, J : float, threshold : float = 0.009) -> tuple:
    """Calculate LR velocity from the correlation data and also from the analytical formula from Wang et al.

    Parameters
    ----------
    correlations : np.ndarray
        correlation data.
    stepsize : float
        step-size of each simulation step.
    lattice_size : int
        Number of sites in the lattice. 
    h : float
        interaction strength with the external field.
    J : float
        coupling strength with the nearest neighbor.
    threshold : float, optional
        threshold for the correlation, by default 0.009

    Returns
    -------
    tuple
        LR velocity : theoretical and extimated.
    """    
    threshold = 0.009

    # Calculate distance to the extreme site from the lattice centre.
    distance = lattice_size // 2

    # Step at which the correlation at the extreme site becomes significant.
    step = np.min(np.where(correlations[:,-1] > threshold))

    # Estimated LR velocity.
    v = distance / (step * stepsize)

    # LR velocity from Wang et al.
    v_paper = 3.02 * np.sqrt(abs(J * h))

    return v, v_paper