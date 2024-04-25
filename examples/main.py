import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

from src.surface import Surface
from src.plot3d import PLOT3D
from src.domain import Domain
from src.wake import Wake
from src.constants import ModelParams
from src.solver import Solver

def main():
    ModelParams.unsteady_problem = False
    ModelParams.static_wake = True
    ModelParams.static_wake_length = 10.0
    # ModelParams.trailing_edge_wake_shed_factor = 1.0
    surface = Surface()

    mesh = PLOT3D()
    filename = 'apame.x'
    mesh.set_surf(surface)
    mesh.read_surface(filename)

    print(f'Mesh has {len(mesh.surf.nodes)} nodes.')
    print(f'Mesh has {len(mesh.surf.panels)} panels.')
    print(f'Panel neighbours: {len(mesh.surf.panel_neighbours)}')

    # mesh.plot_geometry()

    surface.rotate_surface(np.array([0, -5.0, 0.0]), 'deg')
    time_step = 0.5
    fluid_density = 1.225

    # free_stream_vel = np.zeros(3)
    free_stream_vel = np.array([1, 0, 0])
    # surf_vel = np.array([-1, 0, 0])
    # surface.set_linear_vel(surf_vel)
    wake = Wake()
    wake.add_lifting_surf(surface)
    wake.init(free_stream_vel, time_step)
    solver = Solver()
    solver.add_surf(surface)
    solver.add_wake(wake)
    solver.set_fluid_density(fluid_density)
    solver.set_free_stream_vel(free_stream_vel)
    solver.set_ref_vel(free_stream_vel)
    # acceleration = np.zeros([1.2, 0, 0])
    # for i in range(2):
    #     print("Instantaneous surface velocity = ", surf_vel)
    #     solver.set_ref_vel(surf_vel)
    #     solver.solve(time_step, i)
    #     surface.translate_surface(surf_vel * time_step)
    #     solver.convect_wake(time_step)
    #     wake.shed_wake(free_stream_vel, time_step)
    #     surface.set_linear_vel(surf_vel)
    #     print(f"Added mass coefficient = {solver.get_body_forces()[0] / acceleration[0]}")
    
    solver.solve(time_step, 0)
    solver.finalize_iteration()
    print("Body force coeffs: ", solver.get_body_force_coeff())

    print("Program finished.")

if __name__ == "__main__":
    main()