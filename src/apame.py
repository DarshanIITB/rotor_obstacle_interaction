import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

from surface import Surface
from plot3d import PLOT3D
from domain import Domain
from wake import Wake
from constants import ModelParams
from solver import Solver


def main():
    ModelParams.unsteady_problem = False
    ModelParams.static_wake = True
    ModelParams.static_wake_length = 10.0
    surface = Surface()

    mesh = PLOT3D()
    filename = "apame.x"
    mesh.set_surf(surface)
    mesh.read_surface(filename)

    print(f"Mesh has {len(mesh.surf.nodes)} nodes.")
    print(f"Mesh has {len(mesh.surf.panels)} panels.")
    print(f"Panel neighbours: {len(mesh.surf.panel_neighbours)}")

    mesh.plot_geometry()

    surface.rotate_surface(np.array([10, 15.0, 0.0]), "deg")
    time_step = 0.5
    fluid_density = 1.225

    free_stream_vel = np.array([1, 0, 0])
    wake = Wake()
    wake.add_lifting_surf(surface)
    wake.init(free_stream_vel, time_step)
    solver = Solver()
    solver.add_surf(surface)
    solver.add_wake(wake)
    solver.set_fluid_density(fluid_density)
    solver.set_free_stream_vel(free_stream_vel)
    solver.set_ref_vel(free_stream_vel)

    solver.solve(time_step, 0)
    solver.finalize_iteration()
    print("Body force coeffs: ", solver.get_body_force_coeff())

    print("Program finished.")
    solver.plot_cps()


if __name__ == "__main__":
    main()
