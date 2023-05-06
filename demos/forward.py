"""Forward solver example.
"""
import time as tm
import firedrake as fire
import settings
from spyro_ad import io, AcousticSolver, utils
from spyro_ad.io import ensemble_solvers_ad

@ensemble_solvers_ad
def run_forward_true(solver_type, tot_source_num, comm, sn=0):
    """Execute a forward wave propagation.

    Parameters
    ----------
    solver_type : str
        Type of the solver.
    tot_source_num : int
        Total sources number.
    comm: Firedrake.ensemble_communicator
        The MPI communicator for parallelism.
    sn : int, optional
        Source number, by default 0.
    """
    print('######## Running the exact model ########', flush=True)
    p_exact_recv = solver.wave_propagate(
                            vp_exact, source_n=sn,
                            save_rec_data=True
                            )
    if comm.comm.rank == 0:
        io.save_shots(model, comm, p_exact_recv)

vel_model = "circle"
model = settings.model_settings(vel_model)
comm = utils.mpi_init(model)
nshots = model["acquisition"]["num_sources"]
obj = []
mesh, V = io.read_mesh(model, comm)

if vel_model == "circle":
    vp_exact = settings._make_vp_circle(V, mesh, vp_guess=False)  # exact  
elif (vel_model == "marmousi" or vel_model == "br_model"):
    vp_exact = io.interpolate(model["mesh"]["initmodel"], model, mesh, V)  
else:
    raise AssertionError("It is necessary to define the velocity field")              

fire.File("exact_vel.pvd").write(vp_exact)
rec_loc = model["acquisition"]["receiver_locations"]
mesh_rec = fire.VertexOnlyMesh(mesh, rec_loc)
solver = AcousticSolver(model, mesh, mesh_rec)

start = tm.time()
solver_type = "fwd"
tot_sn = len(model["acquisition"]["source_pos"])
for i in range(tot_sn):
    run_forward_true(solver_type, tot_sn, comm, sn=i)
end = tm.time()
print(end-start, flush=True)