import numpy as np
import cupy as cp
import variables as var


def basis_product(flux, basis_arr, axis):
    return cp.tensordot(flux, basis_arr,
                        axes=([axis], [1]))


class DGFlux:
    def __init__(self, resolution, order):
        self.res = resolution
        self.order = order

        # slices into the DG boundaries (list of tuples)
        self.boundary_slices = [(slice(self.res), 0),
                                (slice(self.res), -1)]
        self.boundary_slices_pad = [(slice(self.v_res + 2), 0),
                                    (slice(self.v_res + 2), -1)]
        # self.flux_slice = [(slice(resolution), slice(order))]  # not necessary
        self.num_flux_size = (self.res, 2)

        # for array padding
        self.pad_field, self.pad_spectrum = None, None

        # arrays
        self.flux = var.Scalar(resolution=self.res, order=order)
        self.output = var.Scalar(resolutions=self.res, order=order)

    def semi_discrete_rhs(self, distribution, dielectric, grid):
        """ Computes the semi-discrete equation """
        # Compute the flux
        # self.compute_flux(distribution=distribution, elliptic=elliptic, grid=grid)
        # self.output.arr = (grid.v.J * self.v_flux_lgl(grid=grid, distribution=distribution)

