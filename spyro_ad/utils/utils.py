"""This module contains functions  used in the spyro_ad solvers.
"""
import math
from scipy.signal import butter, filtfilt
import firedrake as fire


def delta_expr(source_loc, mesh, sigma_x=2000.0):
    """Gaussian function.

    Parameters
    ----------
    sigma_x : float, optional
        This parameter , by default 2000.0

    Returns
    -------
    fire.exp
        Guassian function.
    """
    z, x = fire.SpatialCoordinate(mesh)
    x_0 = source_loc[0]
    return fire.exp(-sigma_x * ((z - x_0[0]) ** 2 + (x - x_0[1]) ** 2))


def damping(model, mesh, V):
    """Damping functions for 2D and 3D.

    Parameters
    ----------
    model : dictionary
        Solver settings.
    mesh : fire.Mesh
       Mesh.

    Returns
    -------
    tuple(sigma_x, sigma_z) for 2D.
    tuple(sigma_x, sigma_z, sigma_z) for 3D.
        Damping functions.
    """
    Lx = model["mesh"]["Lx"]
    Lz = model["mesh"]["Lz"]
    a_pml = model["BCs"]["lx"]
    c_pml = model["BCs"]["lz"]
    x1 = 0.0
    x2 = Lx
    z2 = -Lz
    if fire.dimension == 2:
        z, x = fire.SpatialCoordinate(mesh)
    if fire.dimension == 3:
        z, x, y = fire.SpatialCoordinate(mesh)
    damping_type = model["BCs"]["damping_type"]
    if damping_type == "polynomial":
        ps = model["BCs"]["exponent"]  # polynomial scaling
    cmax = model["BCs"]["cmax"]  # maximum acoustic wave velocity
    R = model["BCs"]["R"]  # theoretical reclection coefficient
    bar_sigma = ((3.0 * cmax) / (2.0 * a_pml)) * math.log10(1.0 / R)
    aux1 = fire.Function(V)
    aux2 = fire.Function(V)
    if damping_type != "polynomial":
        fire.warnings.warn(
            "Warning: only polynomial damping functions supported!"
            )
    # Sigma X
    sigma_max_x = bar_sigma  # Max damping
    aux1.interpolate(
        fire.conditional(
            fire.And((x >= x1 - a_pml), x < x1),
            ((abs(x - x1) ** (ps)) / (a_pml ** (ps))) * sigma_max_x,
            0.0,
        )
    )
    aux2.interpolate(
        fire.conditional(
            fire.And(x > x2, (x <= x2 + a_pml)),
            ((abs(x - x2) ** (ps)) / (a_pml ** (ps))) * sigma_max_x,
            0.0,
        )
    )
    sigma_x = fire.Function(V, name="sigma_x").interpolate(aux1 + aux2)

    # Sigma Z
    tol_z = 1.000001
    sigma_max_z = bar_sigma  # Max damping
    aux1.interpolate(
        fire.conditional(
            fire.And(z < z2, (z >= z2 - tol_z * c_pml)),
            ((abs(z - z2) ** (ps)) / (c_pml ** (ps))) * sigma_max_z,
            0.0,
        )
    )

    sigma_z = fire.Function(V, name="sigma_z").interpolate(aux1)

    # sgm_x = File("pmlField/sigma_x.pvd") 
    # sgm_x.write(sigma_x)
    # sgm_z = File("pmlField/sigma_z.pvd")
    # sgm_z.write(sigma_z)

    if fire.dimension == 2:

        return (sigma_x, sigma_z)

    elif fire.dimension == 3:
        Ly = model["mesh"]["Ly"]
        b_pml = model["BCs"]["ly"]
        y1 = 0.0
        y2 = Ly
        # Sigma Y
        sigma_max_y = bar_sigma  # Max damping
        aux1.interpolate(
            fire.conditional(
                fire.And((y >= y1 - b_pml), y < y1),
                ((abs(y - y1) ** (ps)) / (b_pml ** (ps))) * sigma_max_y,
                0.0,
            )
        )
        aux2.interpolate(
            fire.conditional(
                fire.And(y > y2, (y <= y2 + b_pml)),
                ((abs(y - y2) ** (ps)) / (b_pml ** (ps))) * sigma_max_y,
                0.0,
            )
        )
        sigma_y = fire.Function(V, name="sigma_y").interpolate(aux1 + aux2)
        # sgm_y = File("pmlField/sigma_y.pvd")
        # sgm_y.write(sigma_y)

        return (sigma_x, sigma_y, sigma_z)


def mpi_init(model):
    """Initialize computing environment"""
    available_cores = fire.COMM_WORLD.size
    if model["parallelism"]["type"] == "automatic":
        num_cores_per_shot = available_cores / len(model["acquisition"]["source_pos"])
        if available_cores % len(model["acquisition"]["source_pos"]) != 0:
            raise ValueError(
                "Available cores cannot be divided between sources equally."
            )
    elif model["parallelism"]["type"] == "spatial":
        num_cores_per_shot = available_cores
    elif model["parallelism"]["type"] == "custom":
        raise ValueError("Custom parallelism not yet implemented")
    
    comm_ens = fire.Ensemble(fire.COMM_WORLD, num_cores_per_shot)
    return comm_ens


def ricker_wavelet(t, freq, amp=1.0, delay=1.5):
    """Creates a Ricker source function with a
    delay in term of multiples of the distance
    between the minimums.
    """
    t = t - delay * math.sqrt(6.0) / (math.pi * freq)
    return (
        amp
        * (1.0 - (1.0 / 2.0) * (2.0 * math.pi * freq) * (2.0 * math.pi * freq) * t * t)
        * math.exp(
            (-1.0 / 4.0) * (2.0 * math.pi * freq) * (2.0 * math.pi * freq) * t * t
        )
    )


def full_ricker_wavelet(dt, tf, freq, amp=1.0, cutoff=None):
    """Compute the Ricker wavelet optionally applying low-pass filtering
    using cutoff frequency in Hertz.
    """
    nt = int(tf / dt)  # number of timesteps
    time = 0.0
    full_wavelet = np.zeros((nt,))
    for t in range(nt):
        full_wavelet[t] = ricker_wavelet(time, freq, amp)
        time += dt
    if cutoff is not None:
        fs = 1.0 / dt
        order = 2
        nyq = 0.5 * fs  # Nyquist Frequency
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        full_wavelet = filtfilt(b, a, full_wavelet)
    return full_wavelet


def vertex_only_mesh_interpolator(solution, mesh_rec, elastic=False):
    """Vertex only mesh.

    Parameters
    ----------
    solution : FunctionSpace
        A firedrake function space.
    mesh : Firedrake.Mesh
        Firedrake mesh object.
    receiver_locations : numpy array
        Receivers locations.
    elastic : bool
        Option "True" means an elastic wave equation is being executed.
        Option "False" (default) means an acoustic wave equation is being executed.

    Returns
    -------
    tuple[Interpolator, FunctionSpace]
        Firedrake interpolator and function space.
    """
    if elastic:
        P = fire.VectorFunctionSpace(mesh_rec, "DG", 0)
    else:
        P = fire.FunctionSpace(mesh_rec, "DG", 0)

    return fire.Interpolator(solution, P), P