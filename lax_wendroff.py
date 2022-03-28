import numpy as np

def Lax_Wendroff(u_old : np.ndarray, v : float, dx : float, dt : float) -> np.ndarray:
    """Implementation of the Lax-Wendroff Finite Difference scheme
     for the 1D-Advection Equation.

    Parameters
    ----------
    u_old : np.ndarray
        Solution at the start of the time-step.
    v : float
        Advection velocity.
    dx : float
        Spatial grid-size.
    dt : float
        Time-step size.

    Returns
    -------
    np.ndarray
        Solution at the end of the time-step.
    """
    # Calculate the Courant Number and its powers, as per the integration scheme.
     
    courant_number = v * dt / (2.0 * dx)
    courant_number_squared = 2.0 * courant_number**2

    # Calculate the new values of the solution, as per the integration scheme.

    u_new = u_old.copy() 
    u_new[1:-1] = u_old[1:-1] - courant_number * (u_old[2:] - u_old[:-2]) + \
                 courant_number_squared * (u_old[2:] - 2.0 * u_old[1:-1] + u_old[:-2])
    u_new[0]  = u_old[0] - courant_number * (u_old[1] - u_old[-1]) + \
               courant_number_squared * (u_old[1] - 2.0 * u_old[0] + u_old[-1])
    u_new[-1] = u_old[-1] - courant_number * (u_old[0] - u_old[-1]) + \
               courant_number_squared * (u_old[0] - 2.0 * u_old[-1] + u_old[-2])

    return u_new