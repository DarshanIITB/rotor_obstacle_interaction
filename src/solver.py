import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from domain import Domain
from surface import Surface
from wake import Wake
from constants import ModelParams
from utils import set_values


class Solver:
    free_stream_vel = 0
    ref_vel = 0
    density = 0

    def __init__(self):
        self.surf: Surface = None
        self.wake: Wake = None
        self.source_strength: np.ndarray = None
        self.doublet_strength: np.ndarray = None
        self.source_influence: np.ndarray = None
        self.doublet_influence: np.ndarray = None
        self.doublet_influence_mat: np.ndarray = None
        self.RHS = None
        self.solution = None
        self.ksp_doublet = None
        self.surf_vel: np.ndarray = None
        self.pressure_coeffs: np.ndarray = None
        self.surf_pot: np.ndarray = None
        self.surf_pot_old: np.ndarray = None
        self.body_forces = None
        self.body_force_coeffs = None
        self.ref_vel: np.ndarray = None
        self.density = None
        self.wake_doublet_strength: np.ndarray = None
        self.wake_doublet_influence: np.ndarray = None

    def add_surf(self, surface: Surface):
        self.surf = surface

    def add_wake(self, wake: Wake):
        self.wake = wake

    def set_free_stream_vel(self, vel):
        self.free_stream_vel = vel

    def set_ref_vel(self, vel):
        self.ref_vel = vel

    def set_fluid_density(self, rho):
        self.density = rho

    def solve(self, dt, iteration: int):
        # print("ITERATION: ", iteration + 1)
        # print("Computing source strengths...")

        n_panels = self.surf.n_panels()
        self.source_strength = np.zeros(n_panels)
        for p in range(n_panels):
            self.source_strength[p] = self.compute_source_strength(p)
        # print("Done.")
        # print("Computing influence coeff. matrix ...")
        self.source_influence = np.zeros((n_panels, n_panels))
        self.doublet_influence = np.zeros((n_panels, n_panels))

        for n in range(n_panels):
            for p in range(n_panels):
                influence = self.surf.compute_source_doublet_panel_influence(
                    p, self.surf.get_cp(n)
                )
                if p == n:
                    influence[1] = -0.5
                self.source_influence[n][p] = influence[0]
                self.doublet_influence[n][p] = influence[1]

        # print("Done.")

        # print("Applying Kutta condition...")

        if ModelParams.unsteady_problem:
            wake_panel_start = self.wake.n_panels() - self.surf.n_trailing_edge_panels()
            wake_panel_end = self.wake.n_panels()
        else:
            wake_panel_start = 0
            wake_panel_end = self.wake.n_panels()

        for wp in range(wake_panel_start, wake_panel_end):
            TE_panel_counter: int = wp % self.surf.n_trailing_edge_panels()
            upper_panel: int = self.surf.upper_TE_panels[TE_panel_counter]
            lower_panel: int = self.surf.lower_TE_panels[TE_panel_counter]

            for sp in range(n_panels):
                influence = -self.wake.compute_doublet_panel_influence(
                    wp, self.surf.get_cp(sp)
                )

                self.doublet_influence[sp][upper_panel] += influence
                self.doublet_influence[sp][lower_panel] -= influence

        # print("Done.")

        if (
            self.wake_doublet_strength is not None
            and len(self.wake_doublet_strength) > 0
        ):
            # print("Computing wake influence coeffs...")
            self.wake_doublet_influence = np.zeros(
                (n_panels, len(self.wake_doublet_strength))
            )
            for n in range(n_panels):
                for p in range(len(self.wake_doublet_strength)):
                    self.wake_doublet_influence[n][p] = (
                        -self.wake.compute_doublet_panel_influence(
                            p, self.surf.get_cp(n)
                        )
                    )
            # print("Done.")

        # if iteration == 0:
        #     self.initialize
        # print("Solving for unknown doublet strengths...")
        self.setup_linear_system()
        self.solve_linear_system()

        # print("Computing surface velocities...")
        if iteration == 0:
            self.surf_vel = np.zeros((self.surf.n_panels(), 3))

        for p in range(n_panels):
            self.surf_vel[p] = self.compute_surf_vel(p)

        # print("Done.")

        if iteration == 0:
            self.surf_pot = np.zeros(self.surf.n_panels())

        for p in range(n_panels):
            self.surf_pot[p] = self.compute_surf_pot(p)

        # print("Computing pressure...")
        if iteration == 0:
            self.pressure_coeffs = np.zeros(self.surf.n_panels())

        # for p in range(self.surf.n_panels()):
        #     self.pressure_coeffs[p] = self.compute_pressure_coeff(p, iteration, dt)
        # print("Done.")

        if ModelParams.unsteady_problem:
            wake_panel_start = self.wake.n_panels() - self.surf.n_trailing_edge_panels()
            wake_panel_end = self.wake.n_panels()
        else:
            wake_panel_start = 0
            wake_panel_end = self.wake.n_panels()

        wake_doublet_strength = []
        for wp in range(wake_panel_start, wake_panel_end):
            assert len(self.doublet_strength) > 0
            TE_panel_counter = wp % self.surf.n_trailing_edge_panels()
            upper_panel = self.surf.upper_TE_panels[TE_panel_counter]
            lower_panel = self.surf.lower_TE_panels[TE_panel_counter]

            wake_strength = self.doublet_strength[upper_panel]
            wake_doublet_strength.append(wake_strength)
        self.wake_doublet_strength = np.array(wake_doublet_strength)

        self.body_forces = self.compute_body_forces()
        self.body_force_coeffs = self.compute_body_force_coeff()

    def compute_source_strength(self, panel: int):
        node = self.surf.get_cp(panel)
        vel = self.surf.get_velocity(node) - self.free_stream_vel - self.surf.v_ind[panel]
        return vel.dot(self.surf.get_panel_normal(panel))

    def setup_linear_system(self):
        N = self.surf.n_panels()
        col = np.arange(N)
        val = np.zeros(N)
        self.doublet_influence_mat = np.empty((N, N))
        for i in range(N):
            for j in range(N):
                val[j] = self.doublet_influence[i][j]
            set_values(self.doublet_influence_mat, i, N, col, val)

        self.RHS = np.zeros(N)
        for i in range(N):
            for j in range(N):
                self.RHS[i] += self.source_influence[i][j] * self.source_strength[j]

        if (
            self.wake_doublet_influence is not None
            and len(self.wake_doublet_influence) > 0
        ):
            for i in range(N):
                for j in range(len(self.wake_doublet_strength)):
                    self.RHS[i] -= (
                        self.wake_doublet_influence[i][j]
                        * self.wake_doublet_strength[j]
                    )

    def solve_linear_system(self):
        self.solution = np.linalg.solve(self.doublet_influence_mat, self.RHS)
        self.doublet_strength = np.copy(self.solution)
        # print(f"Solution converged in {len(self.solution)} iterations.")

    def compute_surf_vel(self, panel):
        local_vel = (
            self.surf.get_velocity(self.surf.get_cp(panel)) - self.free_stream_vel
        )
        local_vel_transformed = self.surf.transform_vector_panel(panel, local_vel)
        neighbour_panels = self.surf.panel_neighbours[panel]
        neighbour_size = len(neighbour_panels)
        assert neighbour_size > 0
        dim = 2
        rhs = np.zeros(neighbour_size)
        mat = np.zeros((dim, neighbour_size))

        # setup RHS
        for i in range(neighbour_size):
            rhs[i] = (
                self.doublet_strength[neighbour_panels[i]]
                - self.doublet_strength[panel]
            )

        # setup matrix (in column major layout)
        for i in range(len(neighbour_panels)):
            # transform CP of neighbouring node to panel's coordinates
            neighbour_node = self.surf.transform_point_panel(
                panel, self.surf.get_cp(neighbour_panels[i])
            )
            for j in range(dim):
                mat[j, i] = neighbour_node[j]

        res, _, _, _ = lstsq(mat.T, rhs)
        # notice negative sign on rhs terms
        # also notice third component is kept zero
        total_velocity = np.array([-res[0], -res[1], 0]) - np.array(
            [local_vel_transformed[0], local_vel_transformed[1], 0]
        )
        return self.surf.transform_vector_panel_inverse(panel, total_velocity)

    def compute_pressure_coeff(self, panel: int, iteration: int, dt):
        dphi_dt = 0
        if iteration > 0 and ModelParams.unsteady_problem:
            assert dt > 0
            dphi_dt = (self.surf_pot[panel] - self.surf_pot_old[panel]) / dt

        ref_vel = self.free_stream_vel - self.surf.get_velocity(self.surf.get_cp(panel))
        assert np.linalg.norm(self.ref_vel) != 0

        Cp = (
            1.0
            - (np.linalg.norm(self.surf_vel[panel]) ** 2 + 2.0 * dphi_dt)
            / np.linalg.norm(ref_vel) ** 2
        )

        return Cp
    
    def compute_pressure(self, panel:int, iteration: int, dt):
        dphi_dt = 0
        if iteration > 0 and ModelParams.unsteady_problem:
            assert dt > 0
        dphi_dt = (self.surf_pot[panel] - self.surf_pot_old[panel]) / dt
        return (np.linalg.norm(self.surf_vel[panel]) ** 2 + 2.0 * dphi_dt)

    def compute_surf_pot(self, panel: int):
        potential = -self.doublet_strength[panel]
        return potential

    def compute_body_forces(self):
        assert self.density > 0
        force = np.zeros(3)
        for p in range(self.surf.n_panels()):
            ref_vel = self.free_stream_vel - self.surf.get_velocity(self.surf.get_cp(p))
            P_dyn = 0.5 * self.density * np.linalg.norm(ref_vel) ** 2
            force -= self.surf.get_panel_normal(p) * (
                P_dyn * self.pressure_coeffs[p] * self.surf.get_panel_area(p)
            )
        return force

    def compute_body_force_coeff(self):
        P_dyn = 0.5 * self.density * np.linalg.norm(self.ref_vel) ** 2

        planform_area = 0
        for p in range(self.surf.n_panels()):
            planform_area += abs(
                self.surf.get_panel_normal(p)[2] * self.surf.get_panel_area(p)
            )
        planform_area /= 2

        return self.body_forces / (P_dyn * planform_area)

    def compute_total_vel(self, x: np.ndarray):
        assert len(self.source_strength) > 0
        assert len(self.doublet_influence) > 0

        vel = np.zeros(3)

        for sp in range(self.surf.n_panels()):
            vel += (
                self.surf.compute_source_panel_unit_velocity(sp, x)
                * self.source_strength[sp]
            )
            vel += (
                self.surf.compute_doublet_panel_unit_velocity(sp, x)
                * self.doublet_strength[sp]
            )

        for wp in range(len(self.wake_doublet_strength)):
            vel -= (
                self.wake.compute_doublet_panel_unit_velocity(wp, x)
                * self.wake_doublet_strength[wp]
            )

        vel += self.free_stream_vel
        return vel

    def convect_wake(self, dt):
        if not ModelParams.static_wake:
            assert self.wake.n_panels() > 0
            nodes_to_convect = self.wake.n_nodes() - self.surf.n_trailing_edge_nodes()
            assert nodes_to_convect > 0

            wake_vel = np.zeros((nodes_to_convect, 3))

            for wn in range(nodes_to_convect):
                wake_vel[wn] = self.compute_total_vel(self.wake.nodes[wn])
                for wn in range(nodes_to_convect):
                    self.wake.nodes[wn] += wake_vel[wn] * dt

    def compute_domain_vel(self, domain: Domain):
        N = domain.n_nodes()
        assert N > 0
        domain_vel = np.zeros((N, 3))

        for n in range(N):
            vel = self.surf.compute_doublet_panel_unit_velocity(0, domain.nodes[n])
            vel = vel / np.linalg.norm(vel)
            domain_vel[n] = vel

    def write_output(iteration: int):
        pass

    def finalize_iteration(self):
        if not ModelParams.unsteady_problem:
            self.wake_doublet_strength = None

        elif ModelParams.unsteady_problem:
            self.surf_pot_old = self.surf_pot

    def get_body_forces(self):
        return self.body_forces

    def get_body_force_coeff(self):
        return self.body_force_coeffs

    def get_pressure_coeff(self, panel: int):
        return self.pressure_coeffs[panel]

    def plot_cps(self):
        panels = self.surf.panels
        cp_values = self.pressure_coeffs

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Create a polygon for each panel and add it to the plot
        for panel, cp_value in zip(panels, cp_values):
            panel_pts = self.surf.nodes[panel]
            polygon = Poly3DCollection([panel_pts], alpha=0.5)
            polygon.set_facecolor(plt.cm.viridis(cp_value))
            ax.add_collection3d(polygon)

        ax.set_xlim([self.surf.nodes[:, 0].min(), self.surf.nodes[:, 0].max()])
        ax.set_ylim([self.surf.nodes[:, 1].min(), self.surf.nodes[:, 1].max()])
        ax.set_zlim([self.surf.nodes[:, 2].min(), self.surf.nodes[:, 2].max()])

        # Set labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("Surface Cp values")

        # Show colorbar
        sm = plt.cm.ScalarMappable(
            cmap="viridis", norm=plt.Normalize(vmin=min(cp_values), vmax=max(cp_values))
        )
        sm.set_array([])
        fig.colorbar(sm, label="cp value", shrink=0.6)

        plt.show()

    def write_matlab_output(self):
        pass
