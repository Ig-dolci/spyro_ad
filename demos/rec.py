import firedrake as fire
import numpy as np
# mesh = fire.UnitSquareMesh(50, 50)

# rec_loc = np.linspace((0.15, 0.2), (0.15, 0.8), 5)
# mesh_rec = fire.VertexOnlyMesh(mesh, rec_loc)
mesh = fire.UnitSquareMesh(50, 50)
f = mesh.coordinates

f([0.2, 0.4])

