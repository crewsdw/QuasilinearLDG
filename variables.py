import numpy as np
import cupy as cp


class Scalar:
    def __init__(self, resolution, order):
        self.res = resolution
        self.order = order

        # arrays
        self.arr, self.grad = None, None
        self.grad2 = None

    def initialize_particle_distribution(self, grid, v0, vt0, vb, vtb, chi):
        maxwellian0 = grid.compute_maxwellian(thermal_velocity=vt0,
                                              drift_velocity=v0)
        maxwellian1 = grid.compute_maxwellian(thermal_velocity=vtb,
                                              drift_velocity=vb)
        self.arr = (maxwellian0 + chi * maxwellian1) / (1 + chi)

        grad_max0 = grid.compute_maxwellian_gradient(thermal_velocity=vt0,
                                                     drift_velocity=v0)
        grad_max1 = grid.compute_maxwellian_gradient(thermal_velocity=vtb,
                                                     drift_velocity=vb)
        self.grad = (grad_max0 + chi * grad_max1) / (1 + chi)

        grad2_max0 = grid.compute_second_maxwellian_derivative(thermal_velocity=vt0,
                                                               drift_velocity=v0)
        grad2_max1 = grid.compute_second_maxwellian_derivative(thermal_velocity=vtb,
                                                               drift_velocity=vb)
        self.grad2 = (grad2_max0 + chi * grad2_max1) / (1 + chi)

    def initialize_spectral_distribution(self, grid, modes, initial_energy):
        self.arr = cp.zeros_like(grid.arr)
        self.arr[modes] = initial_energy
