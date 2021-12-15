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
        self.grad_spectral = np.tensordot(self.grad, grid.fourier_quads, axes=([0, 1], [1, 2]))

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

    # def cauchy_transform_grad(self, grid):
    #     # for v in self.arr:
    #     vr = np.linspace(-3, 6, num=100)
    #     vi = np.linspace(0.01, 1, num=100)
    #     VR, VI = np.meshgrid(vr, vi, indexing='ij')
    #     Z = np.tensordot(vr, np.ones_like(vi), axes=0) + 1.0j * np.tensordot(np.ones_like(vr), vi, axes=0)
    #
    #     transforms = np.zeros_like(Z) + 0j
    #     transforms = transforms.flatten()
    #     for idxz, z in enumerate(Z.flatten()):
    #         # idx, velocity = grid.get_local_velocity(z)
    #         # print(velocity[0])
    #         # velocity = grid.get_local_velocity(z)[1]
    #         interpolant_cauchy_transform = grid.get_interpolant_cauchy_on_point(z)
    #         # print(interpolant_cauchy_transform.shape)
    #         # print(self.arr.shape)
    #         # quit()
    #         interpolated_cauchy_transform = np.tensordot(self.arr.get() / grid.J_host[:, None],
    #                                                      interpolant_cauchy_transform, axes=([0, 1], [1, 0]))
    #         transforms[idxz] = 1.0 - interpolated_cauchy_transform / (0.25 ** 2.0)
    #     # print(transforms)
    #     transforms = transforms.reshape((100, 100))
    #     plt.figure()
    #     plt.contourf(VR, VI, np.real(Z))
    #     plt.figure()
    #     plt.contourf(VR, VI, np.imag(Z))
    #     plt.figure()
    #     plt.contourf(VR, VI, np.real(transforms))
    #     plt.figure()
    #     plt.contourf(VR, VI, np.imag(transforms))
    #
    #     plt.figure()
    #     plt.contour(VR, VI, np.real(transforms), 0, colors='r')
    #     plt.contour(VR, VI, np.imag(transforms), 0, colors='g')
    #     plt.show()


