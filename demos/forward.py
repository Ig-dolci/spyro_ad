"""Forward solver example.
"""
import time as tm
import numpy as np
import firedrake as fire
import settings
import spyro_ad

# from spyro_ad.solvers.forward_AD import solver_ad

outdir = "fwi/"

vel_model = "circle"
model = settings.model_settings(vel_model)
comm = spyro_ad.utils.mpi_init(model)
nshots = model["acquisition"]["num_sources"]
obj = []
mesh, V = spyro_ad.io.read_mesh(model, comm)

if vel_model == "circle":
    vp_exact = settings._make_vp_circle(V, mesh, vp_guess=False)  # exact  
elif (vel_model == "marmousi" or vel_model == "br_model"):
    vp_exact = spyro_ad.io.interpolate(model["mesh"]["initmodel"], model, mesh, V)  
else:
    raise AssertionError("It is necessary to define the velocity field")              

fire.File("exact_vel.pvd").write(vp_exact)

solver_ad = spyro_ad.AcousticSolver(model, mesh, mesh_rec)

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
        Source number, by default 0
    """
    print('######## Running the exact model ########')
    solver_ad.source_num = sn
    wp = solver_ad.wave_propagate
    output = wp(comm, vp_exact,
                wavelet, output=True, 
                save_rec_data=True
                )
    p_exact_recv = output[0]

    if comm.comm.rank == 0:
        spyro_ad.io.save_shots(
                    model, comm, p_exact_recv
                    )


start = tm.time()
solver_type = "fwd"
tot_sn = len(model["acquisition"]["source_pos"])
for i in range(tot_sn):
    run_forward_true(solver_type, tot_sn, comm, sn=i)
end = tm.time()
print(end-start)