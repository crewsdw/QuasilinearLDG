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
        self.boundary_slices_pad = [(slice(self.res + 2), 0),
                                    (slice(self.res + 2), -1)]
        # self.flux_slice = [(slice(resolution), slice(order))]  # not necessary
        self.num_flux_size = (self.res, 2)

        # for array padding
        self.pad_field, self.pad_spectrum = None, None

        # arrays
        self.flux = var.Scalar(resolution=self.res, order=order)
        self.output = var.Scalar(resolution=self.res, order=order)

    def semi_discrete_rhs(self, distribution, diffusivity, grid):
        """ Computes the semi-discrete equation """
        # Compute the gradient variable
        grad = grid.J[:, None] * self.compute_grad(distribution=distribution, grid=grid)
        # flux = diffusivity * grad
        return grid.J[:, None] * self.compute_dudt(flux=diffusivity * grad, grid=grid)
        # self.compute_flux(distribution=distribution, elliptic=elliptic, grid=grid)
        # self.output.arr = (grid.v.J * self.v_flux_lgl(grid=grid, distribution=distribution)

    def compute_grad(self, distribution, grid):
        """ Compute the gradient variable using one side of the alternating flux """
        return (basis_product(flux=distribution.arr, basis_arr=grid.local_basis.internal, axis=1) -
                self.numerical_flux_grad(distribution=distribution, grid=grid))

    def compute_dudt(self, flux, grid):
        return (basis_product(flux=flux, basis_arr=grid.local_basis.internal, axis=1) -
                self.numerical_flux_arr(flux=flux, grid=grid))

    def numerical_flux_grad(self, distribution, grid):
        # Allocate
        num_flux = cp.zeros(self.num_flux_size) + 0j

        # set padded flux (for rolling, see next operation)
        padded_flux = cp.zeros((self.res + 2, self.order)) + 0j
        padded_flux[1:-1, :] = distribution.arr
        padded_flux[0, -1] = 0.0
        padded_flux[-1, 0] = 0.0

        # "Alternating flux" for gradient: always choose value to the left
        num_flux[self.boundary_slices[0]] = -1.0 * cp.roll(padded_flux[self.boundary_slices_pad[1]],
                                                           shift=+1, axis=1)[1:-1]
        num_flux[self.boundary_slices[1]] = distribution.arr[self.boundary_slices[1]]

        return basis_product(flux=num_flux, basis_arr=grid.local_basis.numerical, axis=2)

    def numerical_flux_arr(self, flux, grid):
        # Allocate
        num_flux = cp.zeros(self.num_flux_size) + 0j

        # set padded flux (for rolling, see next operation)
        padded_flux = cp.zeros((self.res + 2, self.order)) + 0j
        padded_flux[1:-1, :] = flux
        padded_flux[0, -1] = 0.0
        padded_flux[-1, 0] = 0.0

        # "Alternating flux" for basic variable: always choose value to the right
        num_flux[self.boundary_slices[0]] = -1.0 * flux[self.boundary_slices[0]]
        num_flux[self.boundary_slices[1]] = cp.roll(padded_flux[self.boundary_slices_pad[0]],
                                                    shift=-1, axis=1)[1:-1]

        return basis_product(flux=num_flux, basis_arr=grid.local_basis.numerical, axis=2)
