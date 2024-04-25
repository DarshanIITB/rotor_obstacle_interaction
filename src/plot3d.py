import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from surface import Surface
from domain import Domain


class PLOT3D:
    flip_normal = False

    def __init__(self):
        self.surf_filename = ""
        self.domain_filename = ""
        self.surf = None
        self.domain = None
        self.blocks = None
        self.IMAX = None
        self.JMAX = None

    def set_surf_filename(self, filename: str):
        self.surf_filename = filename

    def set_surf(self, surface: Surface):
        self.surf = surface

    def flip_normals(self):
        self.flip_normal = True

    def read_surface(self, filename: str):
        print(f"Reading surface data from {filename}")
        self.set_surf_filename(filename)
        with open(self.surf_filename, "r") as mesh:
            assert mesh

            # Read the number of blocks
            blocks = int(mesh.readline().strip())
            self.blocks = blocks
            assert blocks == 1

            # Read the number of nodes in each direction
            IMAX, JMAX, KMAX = map(int, mesh.readline().split())
            self.IMAX = IMAX
            self.JMAX = JMAX
            assert KMAX == 1

            print(f"Mesh has {(IMAX-1)*(JMAX-1)} panels.")

            # Initialize surface nodes
            self.surf.nodes = np.zeros((IMAX * JMAX, 3))

            # Read coordinates
            nodes_ = []
            for line in mesh:
                values = line.strip().split()
                nodes_.extend(map(float, values))

            cnt = 0
            for dim in range(3):
                for i in range(IMAX):
                    for j in range(JMAX):
                        # Convert 2D index to 1D
                        index = i * JMAX + j
                        self.surf.nodes[index][dim] = nodes_[cnt]
                        cnt += 1

            # Populate panels with node numbers
            panels = []
            # self.surf.panels = np.array([])
            for j in range(JMAX - 1):
                for i in range(IMAX - 1):
                    # plot3d being a structured grid, all panels will have 4 nodes
                    if self.flip_normal:
                        new_panel = [
                            j * IMAX + i,
                            j * IMAX + (i + 1),
                            (j + 1) * IMAX + (i + 1),
                            (j + 1) * IMAX + i,
                        ]
                    else:
                        new_panel = [
                            j * IMAX + (i + 1),
                            j * IMAX + i,
                            (j + 1) * IMAX + i,
                            (j + 1) * IMAX + (i + 1),
                        ]
                    panels.append(new_panel)
            self.surf.panels = np.array(panels)

        # Now 'surface_nodes' and 'surface_panels' contain the data you need
        self.build_topology()

    def build_topology(self):
        imax = self.IMAX - 1
        jmax = self.JMAX - 1
        panel_neighbours = []
        for j in range(jmax):
            for i in range(imax):
                new_neighbours = []
                if i == 0:
                    new_neighbours.append(j * imax + (i + 1))
                elif i == imax - 1:
                    new_neighbours.append(j * imax + (i - 1))
                else:
                    new_neighbours.append(j * imax + (i + 1))
                    new_neighbours.append(j * imax + (i - 1))

                if j == 0:
                    new_neighbours.append((j + 1) * imax + i)
                elif j == jmax - 1:
                    new_neighbours.append((j - 1) * imax + i)
                else:
                    new_neighbours.append((j + 1) * imax + i)
                    new_neighbours.append((j - 1) * imax + i)

                panel_neighbours.append(new_neighbours)

        self.surf.panel_neighbours = np.array(panel_neighbours)

        trailing_edge_nodes = []
        for j in range(self.JMAX):
            for i in range(self.IMAX):
                if i == 0:
                    trailing_edge_nodes.append(j * self.IMAX + i)
        self.surf.trailing_edge_nodes = np.array(trailing_edge_nodes)

        upper_TE_panels = []
        lower_TE_panels = []
        for j in range(jmax):
            for i in range(imax):
                if i == 0:
                    if self.flip_normal:
                        upper_TE_panels.append(j * imax + i)
                    else:
                        lower_TE_panels.append(j * imax + i)
                elif i == imax - 1:
                    if self.flip_normal:
                        lower_TE_panels.append(j * imax + i)
                    else:
                        upper_TE_panels.append(j * imax + i)

        self.surf.upper_TE_panels = np.array(upper_TE_panels)
        self.surf.lower_TE_panels = np.array(lower_TE_panels)

    def set_domain(self, domain: Domain):
        self.domain = domain

    def read_domain(self, filename: str):
        self.domain_filename = filename
        # Read mesh file
        with open(self.domain_filename, "r") as mesh:
            # Assert if mesh file is open
            assert mesh

            # Read the number of blocks
            blocks = int(mesh.readline().strip())
            self.blocks = blocks
            assert blocks == 1

            # Read the number of nodes in each direction
            IMAX, JMAX, KMAX = map(int, mesh.readline().split())
            self.IMAX = IMAX
            self.JMAX = JMAX

            # Set domain ranges
            self.domain.set_domain_ranges(IMAX, JMAX, KMAX)

            # Initialize domain nodes
            self.domain.nodes = np.zeros(IMAX * JMAX * KMAX)

            # Read coordinates (read such that fastest index is x, then y, then z so that it can produce correct vtk output)
            for dim in range(3):
                index = 0
                for k in range(KMAX):
                    for j in range(JMAX):
                        for i in range(IMAX):
                            # Read node value
                            self.domain.nodes[index] = [
                                float(coord) for coord in mesh.readline().split()
                            ]
                            index += 1

    def plot_geometry(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        x = self.surf.nodes[:, 0]
        y = self.surf.nodes[:, 1]
        z = self.surf.nodes[:, 2]
        ax.scatter(x, y, z, c="b", marker="o")

        for panel in self.surf.panels:
            panel_nodes = self.surf.nodes[panel]
            panel_nodes = np.append(
                panel_nodes, [panel_nodes[0]], axis=0
            )  # Close the panel
            ax.plot(panel_nodes[:, 0], panel_nodes[:, 1], panel_nodes[:, 2], c="r")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Plot of Nodes and Panels")

        plt.show()
