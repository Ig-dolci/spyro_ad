"""Settings employed to run the wave equation solvers.
"""
import numpy as np
import firedrake as fire


def model_settings(vel_model: str) -> dict:
    """Settings used to execute the solvers.

    Parameters
    ----------
    vel_model
        Name of the velocity model.
    """
    model = {}

    model["opts"] = {
        "method": "KMV",  # either CG or KMV
        "quadrature": "KMV",  # Equi or KMV
        "degree": 1,  # p order
        "dimension": 2,  # dimension
        "regularization": False,  # regularization is on?
        "gamma": 1e-5,  # regularization parameter
    }

    model["parallelism"] = {
        "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
    }

    # Define the domain size without the ABL.
    if vel_model == "horizont_layers" or vel_model == "circle":
        model["mesh"] = {
            "x0": -0.5,
            "x1": 1.5,
            "z0": 0.0,
            "z1": -1.5,
            "Lz": 1.5,  # depth in km - always positive
            "Lx": 2.0,  # width in km - always positive
            "Ly": 0.0,  # thickness in km - always positive
            "meshfile": "meshes/square.msh",
        }
    if vel_model == "marmousi":
        model["mesh"] = {
            "Lz": 3.5,  # depth in km - always positive
            "Lx": 10.,  # width in km - always positive
            "Ly": 0.0,  # thickness in km - always positive
            "meshfile": "meshes/mm.msh",
            "initmodel": "velocity_models/mm_vp_guess.hdf5",
            "truemodel": "velocity_models/mm_vp.hdf5",
        }
    if vel_model == "br_model":
        model["mesh"] = {
            "Lz": 7.5,  # depth in km - always positive
            "Lx": 17.312,  # width in km - always positive
            "Ly": 0.0,  # thickness in km - always positive
            "meshfile": "meshes/gm.msh",
            #     "initmodel": initmodel + ".hdf5",
            "truemodel": "velocity_models/gm_2020.hdf5",
        }

    # Specify a 250-m Absorbing Boundary Layer (ABL) on the three sides of the domain to damp outgoing waves.
    model["BCs"] = {
        "status": False,  # True or False, used to turn on any type of BC
        "method": "Damping",  # either PML or Damping, used to turn on any type of BC
        "outer_bc": "non-reflective",  # none or non-reflective (outer boundary condition)
        "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
        "exponent": 2,  # damping layer has a exponent variation
        "cmax": 1.5,  # maximum acoustic wave velocity in PML - km/s
        "R": 1e-6,  # theoretical reflection coefficient
        "lz": 1.0,  # thickness of the PML in the z-direction (km) - always positive
        "lx": 1.0,  # thickness of the PML in the x-direction (km) - always positive
        "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
    }
    if vel_model == "horizont_layers" or vel_model == "circle":
        model["acquisition"] = {
            "source_type": "Ricker",
            "frequency": 7.0,
            "delay": 1.0,  # FIXME check this
            "num_sources": 2,  # FIXME not used (remove it, and update an example script)
            # "source_pos": [(-0.11, 0.5)],
            "source_pos": np.linspace((-0.1, 0.2), (-0.1, 0.8), 1),
            "amplitude": 1.0,  # FIXME check this
            "receiver_locations": np.linspace((-0.15, 0.2), (-0.15, 0.8), 10),
        }
    if vel_model == "marmousi" or vel_model == "br_model":
        model["acquisition"] = {
            "source_type": "Ricker",
            "frequency": 7.0,
            "delay": 1.0,
            # "num_sources": 1,
            "num_sources": 1,
            "source_pos": [(-0.125, 5.0)],
            "amplitude": 1.0,
            "num_receivers": 400,
            "receiver_locations": np.linspace((-0.225, 0.2), (-0.225, 9.8), 400),
        }
    
    model["aut_dif"] = {
        "status": True, 
    }

    model["timeaxis"] = {
        "t0": 0.0,  # Initial time for event
        "tf": 1.0,  # Final time for event (for test 7)
        "dt": 0.001,  # timestep size (divided by 2 in the test 4. dt for test 3 is 0.00050)
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "nspool": 200,  # (20 for dt=0.00050) how frequently to output solution to pvds
        "fspool": 1,  # how frequently to save solution to RAM
    }
    return model


def make_vp_circle(V, mesh, vp_guess=False):
    """creating velocity models"""
    x, z = fire.SpatialCoordinate(mesh)
    if vp_guess:
        vp = fire.Function(V).interpolate(1.5 + 0.0 * x)
    else:
        vp = fire.Function(V).interpolate(
            2.5
            + 1 * fire.tanh(100 * (0.125 - fire.sqrt((x + 0.5) ** 2 + (z - 0.5) ** 2)))
        )
    return vp

