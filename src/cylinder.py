import numpy as np
import matplotlib.pyplot as plt
import time

from surface import Surface
from plot3d import PLOT3D
from domain import Domain
from wake import Wake
from constants import ModelParams
from solver import Solver


def main():
    ModelParams.unsteady_problem = True
    ModelParams.trailing_edge_wake_shed_factor = 1.0
    surface = Surface()

    mesh = PLOT3D()
    filename = "cylinder.x"
    mesh.set_surf(surface)
    mesh.read_surface(filename)

    print(f"Mesh has {len(mesh.surf.nodes)} nodes.")
    print(f"Mesh has {len(mesh.surf.panels)} panels.")
    print(f"Panel neighbours: {len(mesh.surf.panel_neighbours)}")

    mesh.plot_geometry()

    surface.rotate_surface(np.array([0, -5.0, 0.0]), "deg")
    time_step = 0.1
    fluid_density = 1.225

    free_stream_vel = np.zeros(3)
    surf_vel = np.array([-1.0, 0.0, 0.0])
    surface.set_linear_vel(surf_vel)

    wake = Wake()
    wake.add_lifting_surf(surface)
    wake.init(free_stream_vel, time_step)

    solver = Solver()
    solver.add_surf(surface)
    solver.add_wake(wake)
    solver.set_fluid_density(fluid_density)

    acceleration = np.array([1.2, 0, 0])
    start_time = time.time()
    for i in range(2):
        print("Instantaneous surface velocity = ", surf_vel)
        solver.set_ref_vel(surf_vel)
        solver.solve(time_step, i)
        surface.translate_surface(surf_vel * time_step)
        solver.convect_wake(time_step)
        wake.shed_wake(free_stream_vel, time_step)
        solver.finalize_iteration()
        surf_vel -= time_step * acceleration
        surface.set_linear_vel(surf_vel)
        print(
            f"Added mass coefficient = {solver.get_body_forces()[0] / acceleration[0]}"
        )

    plt.plot(solver.pressure_coeffs)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Program finished in {execution_time} seconds.")

    solver.plot_cps()


if __name__ == "__main__":
    main()
