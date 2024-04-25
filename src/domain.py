import numpy as np


class Domain:
    def __init__(self):
        self.IMAX, self.JMAX, self.KMAX = None
        self.nodes = np.array([])

    def set_domain_ranges(self, i, j, k):
        self.IMAX = i
        self.JMAX = j
        self.KMAX = k

    def get_IMAX(self):
        return self.IMAX

    def get_JMAX(self):
        return self.JMAX

    def get_KMAX(self):
        return self.KMAX

    def n_nodes(self):
        return len(self.nodes)
