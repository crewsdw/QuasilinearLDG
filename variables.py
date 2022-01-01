import numpy as np
import cupy as cp
import matplotlib.pyplot as plt


class Scalar:
    def __init__(self, resolution, order):
        self.res = resolution
        self.order = order

        # arrays
        self.arr, self.grad = None, None
        self.grad2 = None
        self.arr_spectral, self.grad_spectral = None, None

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

    # NOTE: Try using the QL theory for a distribution on a finite grid, where there is no mode of zero growth rate
    def initialize_spectral_distribution(self, grid, growth_rates, initial_energy):
        self.arr = cp.zeros_like(grid.arr)
        self.arr[growth_rates > 0] = initial_energy * ((growth_rates[growth_rates > 0]) /
                                                       cp.amax(growth_rates[growth_rates > 0]))

        # self.arr = cp.asarray(initial_energy * (growth_rates - cp.amin(growth_rates)) / cp.amax(growth_rates))
        # self.arr[growth_rates < 0] += 1.0e-1 * initial_energy  # * cp.amax(self.arr) * cp.ones_like(grid.arr)
        # self.arr[growth_rates > 0] = initial_energy
        self.arr[:, :] = initial_energy

        plt.figure()
        plt.loglog(grid.arr.flatten(), self.arr.get().flatten(), 'o--')
        plt.xlabel(r'Wavenumber $k\lambda_D$')
        plt.ylabel(r'Initial field power spectrum $\mathcal{E}(k)$')
        plt.grid(True), plt.tight_layout()
        plt.show()

    def compute_grad(self, grid):
        self.grad = cp.tensordot(self.arr,
                                 grid.local_basis.derivative_matrix, axes=([1], [0])) * grid.J[:, None]

    def compute_second_grad(self, grid):
        self.grad2 = cp.tensordot(self.grad,
                                  grid.local_basis.derivative_matrix, axes=([1], [0])) * grid.J[:, None]

    def fourier_transform(self, grid):
        self.arr_spectral = np.tensordot(self.arr, grid.fourier_quads, axes=([0, 1], [1, 2]))

    def fourier_grad(self, grid):
        # self.grad_spectral = np.tensordot(self.grad, grid.fourier_quads, axes=([0, 1], [1, 2]))
        self.grad_spectral = 1j * grid.modes * self.arr_spectral

        # plt.figure()
        # plt.plot(grid.modes.get(), self.grad_spectral.get(), 'o--')
        # plt.show()

    def hilbert_transform_grad(self, grid):
        analytic = cp.sum(2.0 * self.grad_spectral[None, None, :] * grid.grid_phases, axis=2)
        pv_integral = -1.0 * cp.pi * cp.imag(analytic)

        return pv_integral

    def zero_moment(self, grid):
        return cp.tensordot(self.arr,
                            grid.global_quads / grid.J[:, None], axes=([0, 1], [0, 1]))

    def second_moment(self, grid):
        return cp.tensordot(self.arr * (0.5 * grid.device_arr ** 2.0),
                            grid.global_quads / grid.J[:, None], axes=([0, 1], [0, 1]))


class EnergySpectrum:
    def __init__(self):
        self.arr = None

    def initialize_spectral_distribution(self, grid, growth_rates, initial_energy):
        self.arr = cp.zeros_like(grid.device_arr)
        # Like growth rates
        self.arr[growth_rates > 0] = initial_energy * (growth_rates[growth_rates > 0] /
                                                       cp.amax(growth_rates[growth_rates > 0]))
        # gaussian function
        self.arr[:] = initial_energy * cp.exp(-1000 * (grid.device_arr - 0.25) ** 2.0)

    def initialize_spectral_distribution_velocity(self, grid, initial_energy):
        # self.arr = cp.zeros_like(grid.device_arr)
        # gaussian function
        self.arr = initial_energy * cp.exp(-5 * (grid.device_arr - 4.1) ** 2.0)
        # remove parts
        self.arr[grid.device_arr < 2] = 0
        self.arr[grid.device_arr > 10] = 0

    def zero_moment_continuum(self, grid):
        return cp.tensordot(self.arr,
                            grid.global_quads / grid.J[:, None], axes=([0, 1], [0, 1]))

    def zero_moment_continuum_velocity(self, grid):
        integrand = cp.nan_to_num(self.arr / (grid.device_arr ** 2.0))
        integral = cp.tensordot(integrand, grid.global_quads / grid.J[:, None], axes=([0, 1], [0, 1]))
        # print(integral)
        return integral

    def zero_moment_finite_interval(self):
        return self.arr.sum()
