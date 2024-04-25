import numpy as np
from surface import Surface
from constants import ModelParams


class Wake(Surface):
    lifting_surf: Surface = None

    def __init__(self):
        super().__init__()
        self.panels = np.empty((0, 4), dtype=int)
        self.doublet_strength: np.ndarray = None

    def add_lifting_surf(self, surf: Surface):
        self.lifting_surf = surf

    def init(self, free_stream_vels: np.ndarray, dt):
        assert self.lifting_surf != None
        assert self.lifting_surf.n_trailing_edge_nodes() > 0

        self.lifting_surf.update_panel_components()
        N = self.lifting_surf.n_trailing_edge_nodes()
        self.nodes = np.zeros((2 * N, 3))

        for n in range(N):
            node = self.lifting_surf.nodes[self.lifting_surf.trailing_edge_nodes[n]]
            dist = np.zeros(3)
            vel = -self.lifting_surf.get_velocity(node) + free_stream_vels

            if ModelParams.static_wake:
                dist = (
                    self.lifting_surf.get_TE_bisector(n)
                    * ModelParams.static_wake_length
                )
            else:
                dist = vel * dt * ModelParams.trailing_edge_wake_shed_factor

            self.nodes[n] = node + dist

        i: int = 0

        for n in range(N, len(self.nodes)):
            self.nodes[n] = self.lifting_surf.nodes[
                self.lifting_surf.trailing_edge_nodes[i]
            ]
            i += 1

        self.build_topolgy()
        self.update_panel_components()

    def build_topolgy(self):
        assert len(self.nodes) > 0
        spanwise_nodes = self.lifting_surf.n_trailing_edge_nodes()
        spanwise_panels = self.lifting_surf.n_trailing_edge_panels()

        total_nodes = len(self.nodes)
        total_panels = self.n_panels()

        total_new_panels = (
            total_nodes / spanwise_nodes - 1
        ) * spanwise_panels - total_panels

        for p in range(int(total_new_panels)):
            node_a: int = (
                spanwise_nodes + self.n_panels() + self.n_panels() // spanwise_panels
            )
            node_b: int = self.n_panels() + self.n_panels() // spanwise_panels
            node_c: int = node_b + 1
            node_d: int = node_a + 1

            new_panel = np.array([node_a, node_b, node_c, node_d])
            self.panels = np.append(self.panels, [new_panel], axis=0)

    def shed_wake(self, free_stream_vels: np.ndarray, dt: float):
        assert len(self.nodes) > 0

        for n in range(
            len(self.nodes) - self.lifting_surf.n_trailing_edge_nodes(), len(self.nodes)
        ):
            node = self.nodes[n]
            vel = -self.lifting_surf.get_velocity(node) + free_stream_vels

            dist = np.zeros(3)

            if ModelParams.static_wake:
                dist = (
                    self.lifting_surf.get_TE_bisector(n)
                    * ModelParams.static_wake_length
                )
            else:
                dist = vel * dt * ModelParams.trailing_edge_wake_shed_factor
            node += dist

        for i in range(self.lifting_surf.n_trailing_edge_nodes()):
            TE_node = self.lifting_surf.nodes[self.lifting_surf.trailing_edge_nodes[i]]
            np.append(self.nodes, TE_node)

        self.build_topolgy()
        self.update_panel_components()
