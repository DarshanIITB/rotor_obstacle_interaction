import numpy as np
import matplotlib.pyplot as plt
from constants import ModelParams
from math import atan2


class Surface:
    v_lin = np.zeros(3)
    omega = np.zeros(3)
    surf_origin = np.zeros(3)
    surf_orient = np.zeros(3)
    # 1/4pi
    k = 1 / (4 * np.pi)

    def __init__(self):
        self.nodes = np.array([[]])
        self.panels = np.array([[]])
        self.panel_neighbours = np.array([[]])
        self.trailing_edge_nodes = np.array([[]])
        self.upper_TE_panels = np.array([[]])
        self.lower_TE_panels = np.array([[]])
        self.v_ind = np.zeros((self.n_panels(), 3))

    def n_panels(self):
        return len(self.panels)

    def n_nodes(self):
        return len(self.nodes)

    def update_panel_components(self):
        N = self.n_panels()
        assert self.n_nodes() > 0 and N > 0
        self.panel_areas = np.zeros(N)
        self.panel_normals = np.zeros((N, 3))
        self.panel_longitudinals = np.zeros((N, 3))
        self.panel_transverses = np.zeros((N, 3))
        self.cps = np.zeros((N, 3))
        self.panel_local_coords = None
        self.panel_farfield_dist = np.zeros(N)

        # TODO optimization
        panel_local_coords = []
        for p, panel in enumerate(self.panels):
            d1 = 0
            d2 = 0
            # triangular panels
            n = np.zeros(3)
            if len(panel) == 3:
                ab = self.nodes[panel[1]] - self.nodes[panel[0]]
                ac = self.nodes[panel[2]] - self.nodes[panel[0]]
                n = np.cross(ab, ac)
                d1 = np.linalg.norm(ab)
                d2 = np.linalg.norm(ac)
            elif len(panel) == 4:
                ac = self.nodes[panel[2]] - self.nodes[panel[0]]
                bd = self.nodes[panel[3]] - self.nodes[panel[1]]
                n = np.cross(ac, bd)
                d1 = np.linalg.norm(ac)
                d2 = np.linalg.norm(bd)
            self.panel_farfield_dist[p] = max(d1, d2) * ModelParams.farfield_factor
            self.panel_areas[p] = np.linalg.norm(n) / 2.0
            n = n / np.linalg.norm(n)
            self.panel_normals[p] = n

            new_cp = np.zeros(3)
            for i in self.panels[p]:
                new_cp += self.nodes[i]

            new_cp = new_cp / len(self.panels[p])
            self.cps[p] = new_cp

            l = self.nodes[panel[0]] - self.nodes[panel[1]]
            l = l / np.linalg.norm(l)
            self.panel_longitudinals[p] = l / np.linalg.norm(l)
            self.panel_transverses[p] = np.cross(n, l)

            # self.panel_local_coords[p] = np.zeros(len(panel))
            curr_panel_local_coords = []
            for i in range(len(self.panels[p])):
                # self.panel_local_coords[p][i] = self.transform_point_panel(
                #     p, self.nodes[self.panels[p][i]]
                # )
                curr_panel_local_coords.append(
                    self.transform_point_panel(p, self.nodes[self.panels[p][i]])
                )
            panel_local_coords.append(curr_panel_local_coords)
        self.panel_local_coords = np.array(panel_local_coords)

    def transform_to_local_coords(self, panel, vec: np.ndarray):
        l = self.panel_longitudinals[panel]
        n = self.panel_normals[panel]
        t = self.panel_transverses[panel]
        transformed_point = np.array([l, t, n]) @ vec
        return transformed_point

    def transform_to_global_coords(self, panel, vec: np.ndarray):
        l = self.panel_longitudinals[panel]
        n = self.panel_normals[panel]
        t = self.panel_transverses[panel]
        transformed_point = np.array([l, t, n]).T @ vec
        return transformed_point

    def transform_point_panel(self, panel, x):
        cp = self.cps[panel]
        diff = x - cp
        transformed_point = self.transform_to_local_coords(panel, diff)
        return transformed_point

    def get_cp(self, panel):
        assert panel < len(self.panels)
        return self.cps[panel]

    def translate_surface(self, dX):
        for node in self.nodes:
            node += dX

        self.previous_surf_origin = np.copy(self.surf_origin)
        self.surf_origin += dX
        self.update_panel_components()

    def rotate_surface(self, d_theta, units):
        if units != "deg" and units != "rad":
            raise ValueError("units must be 'deg' or 'rad'.")
        d_theta = d_theta * np.pi / 180.0 if units == "deg" else d_theta
        for node in self.nodes:
            x_old = np.copy(node)
            node[0] = (
                np.cos(d_theta[1]) * np.cos(d_theta[2]) * x_old[0]
                + np.cos(d_theta[1]) * np.sin(d_theta[2]) * x_old[1]
                - np.sin(d_theta[1]) * x_old[2]
            )
            node[1] = (
                (
                    np.sin(d_theta[0]) * np.sin(d_theta[1]) * np.cos(d_theta[2])
                    - np.cos(d_theta[0]) * np.sin(d_theta[2])
                )
                * x_old[0]
                + (
                    np.sin(d_theta[0]) * np.sin(d_theta[1]) * np.sin(d_theta[2])
                    + np.cos(d_theta[0]) * np.cos(d_theta[2])
                )
                * x_old[1]
                + np.sin(d_theta[0]) * np.cos(d_theta[1]) * x_old[2]
            )
            node[2] = (
                (
                    np.cos(d_theta[0]) * np.sin(d_theta[1]) * np.cos(d_theta[2])
                    + np.sin(d_theta[0]) * np.sin(d_theta[2])
                )
                * x_old[0]
                + (
                    np.cos(d_theta[0]) * np.sin(d_theta[1]) * np.sin(d_theta[2])
                    - np.sin(d_theta[0]) * np.cos(d_theta[2])
                )
                * x_old[1]
                + np.cos(d_theta[0]) * np.cos(d_theta[1]) * x_old[2]
            )
        self.previous_surf_orient = np.copy(self.surf_orient)
        self.surf_orient += d_theta
        self.update_panel_components()

    def set_linear_vel(self, vel):
        self.v_lin = vel

    def set_v_ind(self, v_ind):
        self.v_ind = v_ind

    def set_omega(self, omega, units):
        if units != "rpm" and units != "rad/s":
            raise ValueError("units must be 'rpm' or 'rad/s'.")
        omega = omega * 2 * np.pi / 60.0 if units == "rpm" else omega

    def n_trailing_edge_nodes(self):
        assert len(self.trailing_edge_nodes) > 0
        return len(self.trailing_edge_nodes)

    def n_trailing_edge_panels(self):
        assert len(self.upper_TE_panels) > 0
        return len(self.upper_TE_panels)

    def get_TE_bisector(self, TE_node: int):
        assert TE_node < self.n_trailing_edge_nodes()
        assert len(self.panel_longitudinals) > 0
        panel = TE_node - 1 if TE_node == self.n_trailing_edge_nodes() - 1 else TE_node
        vec1 = -self.panel_longitudinals[self.lower_TE_panels[panel]]
        vec2 = self.panel_longitudinals[self.upper_TE_panels[panel]]
        bisector = (vec1 + vec2) / 2.0
        bisector = bisector / np.linalg.norm(bisector)
        return bisector

    def get_velocity(self, x: np.ndarray):
        r = self.surf_origin - x
        return self.v_lin + np.cross(self.omega, r)

    def get_panel_normal(self, panel: int):
        return self.panel_normals[panel]

    def compute_surface_normal_influence(self, panel: int, node: np.ndarray):
        transformed_node = self.transform_point_panel(panel, node)
        dist = np.linalg.norm(transformed_node)
        if dist > self.panel_farfield_dist[panel]:
            return self.k * self.panel_areas[panel] / dist

        influence = 0.0
        N = len(self.panels[panel])
        for n in range(N):
            next_node = (n + 1) % N
            node_a = self.panel_local_coords[panel][n]
            node_b = self.panel_local_coords[panel][next_node]
            influence += self.compute_source_edge_influence(
                node_a, node_b, transformed_node
            )
        return -self.k * influence

    def compute_source_edge_influence(
        self, node_a: np.ndarray, node_b: np.ndarray, x: np.ndarray
    ):
        r1 = np.linalg.norm(x - node_a)
        r2 = np.linalg.norm(x - node_b)
        d12 = np.linalg.norm(node_b - node_a)

        if (
            d12 < ModelParams.inversion_tolerance
            or (r1 + r2 - d12) < ModelParams.inversion_tolerance
        ):
            influence = 0
        else:
            influence = (
                (
                    (x[0] - node_a[0]) * (node_b[1] - node_a[1])
                    - (x[1] - node_a[1]) * (node_b[0] - node_a[0])
                )
                / d12
                * np.log((r1 + r2 + d12) / (r1 + r2 - d12))
            )

        if abs(x[2]) > ModelParams.inversion_tolerance:
            e1 = (x[0] - node_a[0]) ** 2 + x[2] ** 2
            e2 = (x[0] - node_b[0]) ** 2 + x[2] ** 2
            h1 = (x[0] - node_a[0]) * (x[1] - node_a[1])
            h2 = (x[0] - node_b[0]) * (x[1] - node_b[1])
            m = (node_b[1] - node_a[1]) / (node_b[0] - node_a[0])
            F = (m * e1 - h1) / (x[2] * r1)
            G = (m * e2 - h2) / (x[2] * r2)
            if F != G:
                influence -= x[2] * atan2(F - G, 1 + F * G)
        return influence

    def compute_doublet_edge_influence(
        self, node_a: np.ndarray, node_b: np.ndarray, x: np.ndarray
    ):
        r1 = np.linalg.norm(x - node_a)
        r2 = np.linalg.norm(x - node_b)
        e1 = (x[0] - node_a[0]) ** 2 + x[2] ** 2
        e2 = (x[0] - node_b[0]) ** 2 + x[2] ** 2
        h1 = (x[0] - node_a[0]) * (x[1] - node_a[1])
        h2 = (x[0] - node_b[0]) * (x[1] - node_b[1])
        m = (node_b[1] - node_a[1]) / (node_b[0] - node_a[0])
        F = (m * e1 - h1) / (x[2] * r1)
        G = (m * e2 - h2) / (x[2] * r2)

        influence = 0.0
        if F != G:
            influence -= x[2] * atan2(F - G, 1 + F * G)
        return influence

    def compute_doublet_panel_influence(self, panel: int, node: np.ndarray):
        transformed_node = self.transform_point_panel(panel, node)
        dist = np.linalg.norm(transformed_node)
        if dist > self.panel_farfield_dist[panel]:
            return self.k * self.panel_areas[panel] * transformed_node[2] * dist**-3.0

        influence = 0.0
        N = len(self.panels[panel])
        for n in range(N):
            next_node = (n + 1) % N
            node_a = self.panel_local_coords[panel][n]
            node_b = self.panel_local_coords[panel][next_node]
            influence += self.compute_doublet_edge_influence(
                node_a, node_b, transformed_node
            )
        return self.k * influence

    def compute_source_doublet_edge_influence(
        self, node_a: np.ndarray, node_b: np.ndarray, x: np.ndarray
    ):
        edge_influence = np.zeros(2)
        r1 = np.linalg.norm(x - node_a)
        r2 = np.linalg.norm(x - node_b)
        d12 = np.linalg.norm(node_b - node_a)
        if (
            d12 < ModelParams.inversion_tolerance
            or (r1 + r2 - d12) < ModelParams.inversion_tolerance
        ):
            edge_influence[0] = 0
        else:
            edge_influence[0] = (
                (
                    (x[0] - node_a[0]) * (node_b[1] - node_a[1])
                    - (x[1] - node_a[1]) * (node_b[0] - node_a[0])
                )
                / d12
                * np.log((r1 + r2 + d12) / (r1 + r2 - d12))
            )

        if abs(x[2]) > ModelParams.inversion_tolerance:
            e1 = (x[0] - node_a[0]) ** 2 + x[2] ** 2
            e2 = (x[0] - node_b[0]) ** 2 + x[2] ** 2
            h1 = (x[0] - node_a[0]) * (x[1] - node_a[1])
            h2 = (x[0] - node_b[0]) * (x[1] - node_b[1])
            m = (node_b[1] - node_a[1]) / (node_b[0] - node_a[0])
            F = (m * e1 - h1) / (x[2] * r1)
            G = (m * e2 - h2) / (x[2] * r2)
            doublet_coeff = 0.0
            if F != G:
                doublet_coeff = atan2(F - G, 1 + F * G)
            edge_influence[0] -= x[2] * doublet_coeff
            edge_influence[1] = doublet_coeff
        return edge_influence

    def compute_source_doublet_panel_influence(self, panel: int, node):
        transformed_node = self.transform_point_panel(panel, node)
        dist = np.linalg.norm(transformed_node)

        influence_coeff = np.zeros(2)
        if dist > self.panel_farfield_dist[panel]:
            influence_coeff[0] = self.k * self.panel_areas[panel] / dist
            influence_coeff[1] = (
                self.k * self.panel_areas[panel] * transformed_node[2] * pow(dist, -3.0)
            )
            return influence_coeff

        N = len(self.panels[panel])
        for n in range(N):
            next_node = (n + 1) % N
            node_a = self.panel_local_coords[panel][n]
            node_b = self.panel_local_coords[panel][next_node]
            edge_influence = self.compute_source_doublet_edge_influence(
                node_a, node_b, transformed_node
            )
            influence_coeff += edge_influence

        influence_coeff[0] *= -self.k
        influence_coeff[1] *= self.k

        return influence_coeff

    def transform_vector_panel_inverse(self, panel: int, x: np.ndarray):
        l = self.panel_longitudinals[panel]
        n = self.panel_normals[panel]
        t = self.panel_transverses[panel]

        transformed_vector = np.array([l, t, n]).T @ x

        return transformed_vector

    def transform_vector_panel(self, panel: int, x: np.ndarray):

        l = self.panel_longitudinals[panel]
        n = self.panel_normals[panel]
        t = self.panel_transverses[panel]

        transformed_vector = np.array([l, t, n]) @ x

        return transformed_vector

    def get_panel_area(self, panel: int):
        return self.panel_areas[panel]

    def compute_source_panel_edge_unit_velocity(
        self, node_a: np.ndarray, node_b: np.ndarray, x: np.ndarray
    ):
        edge_vel = np.zeros(3)
        r1 = np.linalg.norm(x - node_a)
        r2 = np.linalg.norm(x - node_b)
        d12 = np.linalg.norm(node_b - node_a)
        e1 = (x[0] - node_a[0]) ** 2 + x[2] ** 2
        e2 = (x[0] - node_b[0]) ** 2 + x[2] ** 2
        h1 = (x[0] - node_a[0]) * (x[1] - node_a[1])
        h2 = (x[0] - node_b[0]) * (x[1] - node_b[1])
        m = (node_b[1] - node_a[1]) / (node_b[0] - node_a[0])

        if (
            d12 > ModelParams.inversion_tolerance
            and (r1 + r2 - d12) > ModelParams.inversion_tolerance
        ):
            edge_vel[0] = (
                (node_b[1] - node_a[1])
                / d12
                * np.log((r1 + r2 - d12) / (r1 + r2 + d12))
            )
            edge_vel[1] = (
                (node_a[0] - node_b[0])
                / d12
                * np.log((r1 + r2 - d12) / (r1 + r2 + d12))
            )

        F = (m * e1 - h1) / (x[2] * r1)
        G = (m * e2 - h2) / (x[2] * r2)

        if F != G:
            edge_vel[2] = atan2(F - G, 1 + F * G)

        return edge_vel

    def compute_source_panel_unit_velocity(self, panel: int, node: np.ndarray):
        panel_vel = np.zeros(3)
        transformed_node = self.transform_point_panel(panel, node)
        dist = np.linalg.norm(transformed_node)

        if dist > self.panel_farfield_dist[panel]:
            panel_vel = self.panel_areas[panel] * transformed_node * self.k * dist**-3.0
            panel_vel = self.transform_vector_panel_inverse(panel, panel_vel)
            return panel_vel

        N = len(self.panels[panel])
        for n in range(N):
            next_node = (n + 1) % N
            node_a = self.panel_local_coords[panel][n]
            node_b = self.panel_local_coords[panel][next_node]
            panel_vel += self.compute_source_panel_edge_unit_velocity(
                node_a, node_b, transformed_node
            )

        panel_vel = self.transform_vector_panel_inverse(panel, panel_vel)

        return self.k * panel_vel

    def compute_doublet_panel_edge_unit_velocity(
        self, node_a: np.ndarray, node_b: np.ndarray, x: np.ndarray
    ):
        edge_vel = np.zeros(3)
        r1r2x = (x[1] - node_a[1]) * (x[2] - node_b[2]) - (x[2] - node_a[2]) * (
            x[1] - node_b[1]
        )
        r1r2y = -(x[0] - node_a[0]) * (x[2] - node_b[2]) + (x[2] - node_a[2]) * (
            x[0] - node_b[0]
        )
        r1r2z = (x[0] - node_a[0]) * (x[1] - node_b[1]) - (x[1] - node_a[1]) * (
            x[0] - node_b[0]
        )

        r1r2_sq = r1r2x * r1r2x + r1r2y * r1r2y + r1r2z * r1r2z
        r1 = np.linalg.norm(x - node_a)
        r2 = np.linalg.norm(x - node_b)

        if (
            r1 > ModelParams.inversion_tolerance
            and r2 > ModelParams.inversion_tolerance
            and r1r2_sq > ModelParams.inversion_tolerance
        ):
            r0r1 = (
                (node_b[0] - node_a[0]) * (x[0] - node_a[0])
                + (node_b[1] - node_a[1]) * (x[1] - node_a[1])
                + (node_b[2] - node_a[2]) * (x[2] - node_a[2])
            )
            r0r2 = (
                (node_b[0] - node_a[0]) * (x[0] - node_b[0])
                + (node_b[1] - node_a[1]) * (x[1] - node_b[1])
                + (node_b[2] - node_a[2]) * (x[2] - node_b[2])
            )

            Kv = 1.0
            if ModelParams.use_vortex_core_model:
                vm = 1.0
                # decides vortex core model, vm = 1 == rankine model, vm = 2 == scully vortex model
                # h = perpendicular dist of x from line joining x1 and x2,
                # more details: http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html */
                h = np.sqrt(r1r2_sq) / np.sqrt(
                    (node_b[0] - node_a[0]) ** 2
                    + (node_b[1] - node_a[1]) ** 2
                    + (node_b[2] - node_a[2]) ** 2
                )

                rc = 0.003

                # Kv : parameter to disingularize biot savart law,
                # * refer to: Estimating the Angle of Attack from Blade Pressure Measurements on the
                # *           NREL Phase VI Rotor Using a Free Wake Vortex Model: Axial Conditions,
                # *           Wind Energ. 2006; 9:549–577
                Kv = (h * h) / (rc**2 * vm + h**2 * vm) ** 1.0 / vm

            coeff = Kv * self.k / (r1r2_sq) * (r0r1 / r1 - r0r2 / r2)

            edge_vel[0] = r1r2x * coeff
            edge_vel[1] = r1r2y * coeff
            edge_vel[2] = r1r2z * coeff
        return edge_vel

    def compute_doublet_panel_unit_velocity(self, panel: int, node: np.ndarray):
        panel_vel = np.zeros(3)
        N = len(self.panels[panel])
        for n in range(N):
            next_node = (n + 1) % N
            node_a = self.panel_local_coords[panel][n]
            node_b = self.panel_local_coords[panel][next_node]
            panel_vel -= self.compute_doublet_panel_edge_unit_velocity(
                node_a, node_b, node
            )
        return panel_vel
