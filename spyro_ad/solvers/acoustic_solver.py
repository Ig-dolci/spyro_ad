"""This module provides acoustic solver.
Can execute forward and implemented adjoint solvers.
"""
import numpy as np
from .. import utils
from ..domains import quadrature, space
import firedrake as fire
fire.set_log_level(fire.ERROR)

__all__ = [
    "AcousticSolver",
]


class AcousticSolver():
    """Secord-order in time fully-explicit scheme.

        CG FEM with or without higher order mass lumping (KMV type elements).
    
        Atributes
        ----------
        model: dict
            Contains model options and parameters
        mesh: Firedrake.mesh object
            The 2D/3D spatial triangular mesh.
        mesh_rec: Firedrake.mesh object
            The 2D/3D receiver mesh.
        """
    def __init__(
            self, model, mesh, mesh_rec, solver="fwd"
            ):
        
        self.mesh = mesh
        self.model = model
        self.mesh_rec = mesh_rec
        self.solver = solver
        self.tolerance = 0.0000001
        self.J0 = 0

    def wave_propagate(self, c, source_n=0, compute_funct=False, output=False, **kwargs):
        """Acoustic wave equation solver.

        Parameters
        ----------
        comm: Firedrake.ensemble_communicator
            The MPI communicator for parallelism.
        c: Firedrake.Function
            The velocity model interpolated onto the mesh.
        output: `boolean`, optional
            Whether or not to write results to pvd files.

        Returns
        -------
        usol: list of Firedrake.Functions
            The full field solution at `fspool` timesteps
        usol_recv: array-like
            The solution interpolated to the receivers at all timesteps
        J0: Cost function associated to a single source 
        dJ: Gradient field associate to a single source when the
            implemented adjoint is employed
        misfit: The misfit function
        """
        model = self.model
        method = model["opts"]["method"]
        degree = model["opts"]["degree"]
        dim = model["opts"]["dimension"]
        dt = model["timeaxis"]["dt"]
        tf = model["timeaxis"]["tf"]
        nt = int(tf / dt)  # number of timesteps
        bc = model["BCs"]["status"]
        freq = model["acquisition"]["frequency"]
        source_loc = model["acquisition"]["source_pos"][source_n]
        element = space.FE_method(self.mesh, method, degree)
        V = fire.FunctionSpace(self.mesh, element)
        # kwargs
 
        save_rec_data = kwargs.get("save_rec_data")
        save_p = kwargs.get("save_p")
        p_true_rec = kwargs.get("p_true_rec")
        params = self.params()
        qr_x, qr_s, _ = quadrature.quadrature_rules(V)
           
        u = fire.TrialFunction(V)
        v = fire.TestFunction(V)  # Test Function
        f = fire.Function(V, name="f")
        X = fire.Function(V, name="X")
        u_n = fire.Function(V, name="u_n")      # n
        u_nm1 = fire.Function(V, name="u_nm1")      # n-1

        du2_dt2 = ((u - 2.0 * u_n + u_nm1) / fire.Constant(dt ** 2))
        t_term = du2_dt2 * v * fire.dx(scheme=qr_x)
        l_term = c * c * fire.dot(fire.grad(u_n), fire.grad(v)) * fire.dx(scheme=qr_x)
        f_term = f * v * fire.dx(scheme=qr_x)
        nf = 0
        if model["BCs"]["outer_bc"] == "non-reflective":
            nf = c * ((u_n - u_nm1) / dt) * v * fire.ds(scheme=qr_s)    
        FF = t_term + l_term - f_term + nf
        
        if bc:
            if dim == 2:
                self.damp_conditions(u_n, u_nm1, v, u, qr_x, V)
            if dim == 3:
                self.damp_conditions(u_n, u_nm1, v, u, qr_x, V)

        lhs_ = fire.lhs(FF)
        rhs_ = fire.rhs(FF)

        lin_var = fire.LinearVariationalProblem(lhs_, rhs_, X)
        solver = fire.LinearVariationalSolver(lin_var, solver_parameters=params)
        usol_recv = []
        usol = []
        misfit = []

        wavelet = utils.full_ricker_wavelet(dt, tf, freq)
        rec_interp, P = utils.vertexonlymesh_interpolator(u_nm1, self.mesh_rec)
        g = fire.Function(V).interpolate(utils.delta_expr(source_loc, self.mesh))
        for step in range(nt):
            if self.solver == "bwd":
                fn = fire.Function(V)
                f.assign(fn)
               
            else:
                # f_temp = fire.Function(V)
                # f.dat.data[:] = 1.0
                w = fire.Constant(wavelet[step])
                f.assign(g*w)
                # f = self.excitation.apply_source(f, wavelet[step]/0.0001)
                # f.interpolate(f_temp)
            solver.solve()
            u_nm1.assign(u_n)
            u_n.assign(X)
            if self.solver == "fwd":
                rec = fire.Function(P, name="rec")
                rec_interp.interpolate(output=rec)
                if save_rec_data:
                    usol_recv.append(rec.vector().gather())
                if compute_funct:
                    true_rec = fire.Function(P, name="true_rec")
                    true_rec.dat.data[:] = p_true_rec[step]
                    misfit = rec - true_rec
                    J = fire.assemble(0.5*fire.inner(misfit, misfit) * fire.dx)
                    self.J0 += J

        fire.File("u.pvd").write(X)
        if save_rec_data:
            return usol_recv
        if save_p:
            return usol_recv, usol
    
    def params(self) -> set:
        """Element parameters.

        Raises
        ------
        ValueError
            Method is not yet supported.
        """
        method = self.model["opts"]["method"]
        if method == "KMV":
            params = {
                    "mat_type": "matfree", "ksp_type": 
                    "preonly", "pc_type": "jacobi"
                    }
        elif (
            method == "CG"
            and fire.mesh.ufl_cell() != fire.quadrilateral
            and fire.mesh.ufl_cell() != fire.hexahedron
        ):
            params = {
                    "mat_type": "matfree", "ksp_type": "cg",
                    "pc_type": "jacobi"
                    }
        elif method == "CG" and (
            fire.mesh.ufl_cell() == fire.quadrilateral
            or fire.mesh.ufl_cell() == fire.hexahedron
        ):
            params = {
                    "mat_type": "matfree", "ksp_type": "preonly",
                    "pc_type": "jacobi"
                    }
        else:
            raise ValueError("Method is not yet supported.")  
        return params
        
    def damp_conditions(self, u_n, u_nm1, v, u, qr_x, V):
        """Set the boundary conditions.

        Parameters
        ----------
        c : _type_
            _description_
        u_n : _type_
            _description_
        u_nm1 : _type_
            _description_
        v : _type_
            _description_
        u : _type_
            _description_
        qr_s : _type_
            _description_
        qr_x : _type_
            _description_
        """
        model = self.model
        dim = model["opts"]["dimension"]
        dt = model["timeaxis"]["dt"]
        if dim == 2:
            sigma_x, sigma_z = utils.damping(model, self.mesh, V)
            return (
                    (sigma_x + sigma_z)
                    * ((u - u_nm1) / fire.Constant(2.0 * dt))
                    * v * fire.dx(scheme=qr_x)
                )
        elif dim == 3:
            sigma_x, sigma_y, sigma_z = utils.damping(model, self.mesh, V)
         
            return (
                    (sigma_x + sigma_y + sigma_z)
                    * ((u - u_n) / fire.Constant(dt))
                    * v * fire.dx(scheme=qr_x)
                    )

    def grad_solver(self, c, qr_x, method, V):
        """Set the gradient solver.

        Parameters
        ----------
        c : firedrake.Function
            P-speed parameter.
        qr_x : _type_
            _description_
        method : str
            _description_

        Returns
        -------
        _type_
            _description_
        """
        m_u = fire.TrialFunction(V)
        m_v = fire.TestFunction(V)
        mgrad = m_u * m_v * fire.dx(scheme=qr_x)

        uuadj = fire.Function(V)  # auxiliarly function for the gradient compt.
        uufor = fire.Function(V)  # auxiliarly function for the gradient compt.

        ffG = 2.0 * c * fire.dot(
                    fire.grad(uuadj), fire.grad(uufor)
                    ) * m_v * fire.dx(scheme=qr_x)

        G = mgrad - ffG
        lhsG, rhsG = fire.lhs(G), fire.rhs(G)

        gradi = fire.Function(V)
        grad_prob = fire.LinearVariationalProblem(lhsG, rhsG, gradi)
        if method == "KMV":
            grad_solver = fire.LinearVariationalSolver(
                grad_prob,
                solver_parameters={
                    "ksp_type": "preonly",
                    "pc_type": "jacobi",
                    "mat_type": "matfree",
                },
            )
        elif method == "CG":
            grad_solver = fire.LinearVariationalSolver(
                grad_prob,
                solver_parameters={
                    "mat_type": "matfree",
                },
            )
        return grad_solver, uuadj, uufor, gradi
    