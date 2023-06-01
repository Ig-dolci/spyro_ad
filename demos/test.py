import math
import firedrake as fire
import firedrake_adjoint as fire_adj
import numpy as np
from scipy.signal import butter, filtfilt
mesh = fire.UnitSquareMesh(50, 50)

element = fire.FiniteElement("KMV", mesh.ufl_cell(), degree=1, variant="KMV")
V = fire.FunctionSpace(mesh, element)
rec_loc = np.linspace((0.5, 0.2), (0.5, 0.8), 4)
mesh_rec = fire.VertexOnlyMesh(mesh, rec_loc)
source_loc = np.linspace((0.1, 0.2), (0.1, 0.8), 1)

tf = 1.0
freq = 7
dt = 0.001
nt = int(tf/dt)


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


def _make_vp_circle(vp_guess=False):
    """creating velocity models"""
    x, z = fire.SpatialCoordinate(mesh)
    if vp_guess:
        vp = fire.Function(V).interpolate(1.5 + 0.0 * x)
    else:
        vp = fire.Function(V).interpolate(
            2.5
            + 1 * fire.tanh(100 * (0.125 - fire.sqrt((x - 0.5) ** 2 + (z - 0.5) ** 2)))
        )
    fire.File("vp.pvd").write(vp)
 
    return vp


def forward_solver(c):
    f = fire.Function(V)
    u = fire.TrialFunction(V)
    v = fire.TestFunction(V)  # Test Function
    f = fire.Function(V, name="f")
    X = fire.Function(V, name="X")
    u_n = fire.Function(V, name="u_n")      # n
    u_nm1 = fire.Function(V, name="u_nm1")      # n-1
    du2_dt2 = ((u - 2.0 * u_n + u_nm1) / fire.Constant(dt ** 2))
    t_term = du2_dt2 * v * fire.dx
    l_term = c * c * fire.dot(fire.grad(u_n), fire.grad(v)) * fire.dx
    f_term = f * v * fire.dx
    nf = c * ((u_n - u_nm1) / dt) * v * fire.ds  
    FF = t_term + l_term - f_term + nf
    lhs_ = fire.lhs(FF)
    rhs_ = fire.rhs(FF)
    lin_var = fire.LinearVariationalProblem(lhs_, rhs_, X)
    params = {
            "mat_type": "matfree", "ksp_type": 
            "preonly", "pc_type": "jacobi"
            }
    wavelet = full_ricker_wavelet(dt, tf, freq)
    solver = fire.LinearVariationalSolver(lin_var, solver_parameters=params)
    z, x = fire.SpatialCoordinate(mesh)
    x_0 = source_loc
    delta = fire.exp(-2000 * ((z - x_0[0]) ** 2 + (x - x_0[1]) ** 2))
    g = fire.Function(V).interpolate(delta)
    P = fire.FunctionSpace(mesh_rec, "DG", 0)
    rec_interp = fire.Interpolator(X, P)
    usol_recv = []
    for step in range(nt):
        w = fire.Constant(wavelet[step])
        f.assign(g*w)
        solver.solve()
        u_nm1.assign(u_n)
        u_n.assign(X)
        rec = fire.Function(P, name="rec")
        rec_interp.interpolate(output=rec)
        usol_recv.append(rec.vector().gather())


c = _make_vp_circle()
P = fire.FunctionSpace(mesh_rec, "DG", 0)
rec_interp = fire.Interpolator(c, P)
rec = fire.Function(P, name="rec")
rec_interp.interpolate(output=rec)
forward_solver(c)