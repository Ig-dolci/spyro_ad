import time as tm
from mpi4py import MPI
from scipy import optimize
import numpy as np
import firedrake as fire
import firedrake_adjoint as fire_adj
import settings
import spyro_ad
# from spyro_ad import io, AcousticSolver, utils
# from spyro_ad.io import ensemble_solvers_ad

outdir = "fwi/"
global nshots

vel_model = "circle"
model = settings.model_settings(vel_model)
comm = spyro_ad.utils.mpi_init(model)
nshots = model["acquisition"]["num_sources"]  
obj = []

mesh, V = spyro_ad.io.read_mesh(model, comm)
p_exact_recv = []

if vel_model == "circle":
    vp_guess = settings.make_vp_circle(V, mesh, vp_guess=True)

if (vel_model == "marmousi" or vel_model == "br_model"):
    vp_guess = spyro_ad.io.interpolate(model["mesh"]["initmodel"], model, mesh, V)
   
# if comm.ensemble_comm.rank == 0:
#     control_file = fire.File(outdir + "control.pvd", comm=comm.comm)
#     grad_file = fire.File(outdir + "grad.pvd", comm=comm.comm)

rec_loc = model["acquisition"]["receiver_locations"]
mesh_rec = fire.VertexOnlyMesh(mesh, rec_loc)
solver = spyro_ad.AcousticSolver(model, mesh, mesh_rec)


# @ensemble_solvers_ad
def runfwi(solver_type, tot_source_num, comm, xi, sn=0):
    """Execute an acoustic FWI problem.

    Parameters
    ----------
    solver_type : str
        Type of solver: either forward or backward.
    tot_source_num : int
        Total source number.
    comm: Firedrake.ensemble_communicator
        The MPI communicator for parallelism
    xi : numpy array
        Output of the optimization method which is the updated velocity model.
    sn : int, optional
        Source number, by default 0

    Returns
    -------
    float, numpy array
        Objective functional and adjoint-based gradient.
    """
    aut_dif = model["aut_dif"]["status"]
    
    local_mesh_index = mesh.coordinates.node_set.halo.local_to_global_numbering
    vp_guess = spyro_ad.utils.scatter_data_function(xi, V, comm, local_mesh_index, name="vp_guess")
    if comm.ensemble_comm.rank == 0:
        fire.File("vp_guess.pvd", comm=comm.comm).write(vp_guess)
    
    print('######## Running the guess model ########')
    Jm = solver.wave_propagate(
                vp_guess, source_n=sn, p_true_rec=p_exact_recv[sn],
                compute_funct=True
                )
    control = fire_adj.Control(vp_guess)
    dJ = fire_adj.compute_gradient(Jm, control)
    fire_adj.get_working_tape().clear_tape()
    if comm.ensemble_comm.rank == 0:
        fire.File("grad.pvd", comm=comm.comm).write(dJ)
    return Jm, dJ


def exec_fwd_source(xi):
    """Execute the forward solver for n solver.

    Parameters
    ----------
    xi : like array
        Control parameter.

    Returns
    -------
    _type_
        _description_
    """
    solver_type = "fwi"
    J_total = 0
