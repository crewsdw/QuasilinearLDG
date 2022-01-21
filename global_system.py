import numpy as np
import cupy as cp
import variables as var
import matplotlib.pyplot as plt


class Global:
    """ Class to build global matrix system of DG based on fluxes """
    def __init__(self, grid):
        self.res, self.order = grid.elements, grid.order
        self.size = self.res * self.order
        self.matrix = None
        self.inv_matrix = None

        self.indicator_dist = var.Scalar(resolution=self.res, order=self.order)

    def build_global_system(self, diffusivity, flux, grid):
        self.matrix = cp.zeros((self.size, self.size))

        # Build matrix one row at a time
        indicator = cp.zeros(self.size)
        for idx in range(self.size):
            # set indicator
            indicator[idx] = 1
            self.indicator_dist.arr = indicator.reshape((self.res, self.order))
            # get rhs
            flux2, du_dt = flux.semi_discrete_rhs(distribution=self.indicator_dist, diffusivity=diffusivity, grid=grid)
            self.matrix[:, idx] = du_dt.flatten()
            # re-set indicator
            indicator[idx] = 0

        # self.inv_matrix = cp.linalg.inv(self.matrix)
