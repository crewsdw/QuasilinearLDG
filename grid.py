import numpy as np
import cupy as cp
import basis as b
import matplotlib.pyplot as plt
import scipy.special as sp


# import tools.plasma_dispersion as pd


class VelocityGrid:
    """ In this experiment, the velocity grid is an LGL quadrature grid """

    def __init__(self, low, high, elements, order):
        self.low, self.high = low, high
        self.elements, self.order = elements, order
        self.local_basis = b.LGLBasis1D(order=self.order)
        # self.local_basis = b.GLBasis1D(order=self.order)

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # arrays
        self.arr, self.device_arr = None, None
        self.mid_points = None
        self.create_even_grid()

        # stretch / transform elements
        # self.pole_distance = 5
        self.dx_grid = None
        self.stretch_grid()
        # self.modes = 2.0 * np.pi * np.arange(1 - int(2 * self.elements), int(2 * self.elements)) / self.length
        self.modes = 2.0 * np.pi / self.length * cp.arange(int(self.elements // 2))
        self.element_idxs = np.arange(self.elements)

        # jacobian
        # self.J = 2.0 / self.dx
        self.J = cp.asarray(2.0 / self.dx_grid)
        self.J_host = self.J.get()
        # plt.figure()
        # x = np.linspace(-500, 500, num=5)
        # X, V = np.meshgrid(x, self.arr.flatten(), indexing='ij')
        # plt.plot(X, V, 'ko--')
        # plt.plot(X.T, V.T, 'ko--')
        # for i in range(self.elements):
        #     plt.plot(np.zeros_like(self.arr[i, :]), self.arr[i, :], 'ko')
        # plt.show()

        # global quad weights
        self.global_quads = cp.tensordot(cp.ones(elements),
                                         cp.asarray(self.local_basis.weights), axes=0)

        # global translation matrix
        mid_identity = np.tensordot(self.mid_points, np.eye(self.local_basis.order), axes=0)
        self.translation_matrix = cp.asarray(mid_identity +
                                             self.local_basis.translation_matrix[None, :, :] /
                                             self.J[:, None, None].get())

        # quad matrix
        self.fourier_quads = cp.asarray((self.local_basis.weights[None, None, :] *
                                         np.exp(-1j * self.modes[:, None, None].get() * self.arr[None, :, :]) /
                                         self.J[None, :, None].get()) / self.length)
        self.grid_phases = cp.exp(1j * self.modes[None, None, :] * self.device_arr[:, :, None])

    def create_even_grid(self):
        """ Build global grid """
        # translate to [0, 1]
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        # element left boundaries (including ghost elements)
        xl = np.linspace(self.low, self.high - self.dx, num=self.elements)
        # construct coordinates
        self.arr = np.zeros((self.elements, self.order))
        for i in range(self.elements):
            self.arr[i, :] = xl[i] + self.dx * np.array(nodes_iso)
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])

    def stretch_grid(self):
        # Investigate grid mapping
        alphas, betas = (np.array([0.64, 0.64, 0.64, 0.64]),  # , 0.64]),
                         np.array([0.5, 0.5, 0.5, 0.5]))  # , 0.5]))
        # plt.figure()
        # plt.plot(self.arr.flatten(), self.iterate_map_asym(self.arr, alphas[:-2], betas[:-2]).flatten(),
        #          'k', label=r'Iteration 1')
        # plt.plot(self.arr.flatten(), self.iterate_map_asym(self.arr, alphas[:-1], betas[:-1]).flatten(),
        #          'k', label=r'Iteration 2')
        # plt.plot(self.arr.flatten(), self.iterate_map_asym(self.arr, alphas, betas).flatten(),
        #          'r', label=r'Iteration 3')
        # plt.xlabel('Input points'), plt.ylabel('Output points')
        # plt.grid(True), plt.legend(loc='best'), plt.tight_layout()
        # plt.show()
        # Map points
        # Map lows and highs
        # orders = [0.4, 0.5, 0.8, 0.8, 0.8]  # , 0.8, 0.8]
        # mapped_lows = self.iterate_map(self.arr[:, 0], orders=orders)
        # mapped_highs = self.iterate_map(self.arr[:, -1], orders=orders)
        # alphas, betas = np.array([0.3, 0.55]), np.array([0.35, 0.6])
        mapped_lows = self.iterate_map_asym(self.arr[:, 0], alphas=alphas, betas=betas)
        mapped_highs = self.iterate_map_asym(self.arr[:, -1], alphas=alphas, betas=betas)
        self.dx_grid = mapped_highs - mapped_lows
        # self.dx_grid = self.dx * self.grid_map(self.mid_points)  # mapped_highs - mapped_lows
        # print(self.dx_grid)
        # xl = np.linspace(self.low, self.high - self.dx, num=self.elements)
        # Overwrite coordinate array
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        lows = np.zeros(self.elements + 1)
        lows[0] = self.low
        for i in range(self.elements):
            self.arr[i, :] = lows[i] + self.dx_grid[i] * np.array(nodes_iso)
            lows[i + 1] = self.arr[i, -1]
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])

    def iterate_map(self, points, orders):
        for order in orders:
            points = self.grid_map(points, order)
        return points

    def iterate_map_asym(self, points, alphas, betas):
        for idx, alpha in enumerate(alphas):
            points = self.grid_map_asym(points, alpha, betas[idx])
        return points

    def grid_map_asym(self, points, alpha, beta):
        return (self.low * ((self.high - points) / self.length) ** alpha +
                self.high * ((points - self.low) / self.length) ** beta)

    def grid_map(self, points, order):
        return (self.low * ((self.high - points) / self.length) ** order +
                self.high * ((points - self.low) / self.length) ** order)

    def compute_maxwellian(self, thermal_velocity, drift_velocity):
        return cp.exp(-0.5 * ((self.device_arr - drift_velocity) /
                              thermal_velocity) ** 2.0) / (np.sqrt(2.0 * np.pi) * thermal_velocity)

    def compute_maxwellian_gradient(self, thermal_velocity, drift_velocity):
        return (-1.0 * ((self.device_arr - drift_velocity) / thermal_velocity ** 2.0) *
                self.compute_maxwellian(thermal_velocity=thermal_velocity, drift_velocity=drift_velocity))

    def compute_second_maxwellian_derivative(self, thermal_velocity, drift_velocity):
        return -self.compute_maxwellian(thermal_velocity=thermal_velocity, drift_velocity=drift_velocity) + \
               -1.0 * ((self.device_arr - drift_velocity) / thermal_velocity ** 2.0) * \
               self.compute_maxwellian_gradient(thermal_velocity=thermal_velocity, drift_velocity=drift_velocity)

    def get_local_velocity(self, phase_velocity):
        # find out which element the phase velocity is in
        idx = self.element_idxs[(self.arr[:, 0] < np.real(phase_velocity)) &
                                (np.real(phase_velocity) < self.arr[:, -1])]
        # get local velocity
        velocity = self.J_host[idx] * (phase_velocity - self.mid_points[idx])

        return idx, velocity

    def get_interpolant_on_point(self, velocity):
        # get interpolation polynomial on point
        vandermonde_on_point = np.array([sp.legendre(s)(velocity)
                                         for s in range(self.order)])
        interpolant_on_point = np.tensordot(self.local_basis.inv_vandermonde,
                                            vandermonde_on_point, axes=([0], [0]))
        return interpolant_on_point

    def get_interpolant_grad_on_point(self, velocity):
        # get interpolation polynomial on point
        vandermonde_grad_on_point = np.array([sp.legendre(s).deriv()(velocity)
                                              for s in range(self.order)])
        interpolant_grad_on_point = np.tensordot(self.local_basis.inv_vandermonde,
                                                 vandermonde_grad_on_point, axes=([0], [0]))
        return interpolant_grad_on_point

    def get_interpolant_cauchy_on_point(self, velocity):
        # transform velocity to each element's coordinates
        velocities = self.J_host * (velocity - self.mid_points)

        # get cauchy transform on point
        cauchy_vandermonde_grad_on_point = 2.0 * np.array([[sp.lqmn(0, s, velocities[m])[0][0][0]
                                                           for m in range(self.elements)]
                                                        for s in range(self.order)])
        cauchy_grad_on_point = np.tensordot(self.local_basis.inv_vandermonde,
                                            cauchy_vandermonde_grad_on_point, axes=([0], [0]))
        return cauchy_grad_on_point


class SpectralGrid:
    """ In this experiment, the velocity grid is an LGL quadrature grid """

    def __init__(self, low, high, elements, order):
        self.low, self.high = low, high
        self.elements, self.order = elements, order
        self.local_basis = b.LGLBasis1D(order=self.order)
        # self.local_basis = b.GLBasis1D(order=self.order)

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # arrays
        self.arr, self.device_arr = None, None
        self.mid_points = None
        self.create_even_grid()

        # stretch / transform elements
        # self.pole_distance = 5
        self.dx_grid = None
        self.stretch_grid()
        # self.modes = 2.0 * np.pi * np.arange(1 - int(2 * self.elements), int(2 * self.elements)) / self.length
        self.modes = 2.0 * np.pi / self.length * np.arange(int(self.elements))
        self.element_idxs = np.arange(self.elements)

        # jacobian
        # self.J = 2.0 / self.dx
        self.J = cp.asarray(2.0 / self.dx_grid)
        # plt.figure()
        # x = np.linspace(-500, 500, num=5)
        # X, V = np.meshgrid(x, self.arr.flatten(), indexing='ij')
        # plt.plot(X, V, 'ko--')
        # plt.plot(X.T, V.T, 'ko--')
        # for i in range(self.elements):
        #     plt.plot(np.zeros_like(self.arr[i, :]), self.arr[i, :], 'ko')
        # plt.show()

        # global quad weights
        self.global_quads = cp.tensordot(cp.ones(elements),
                                         cp.asarray(self.local_basis.weights), axes=0)

        # global translation matrix
        mid_identity = np.tensordot(self.mid_points, np.eye(self.local_basis.order), axes=0)
        self.translation_matrix = cp.asarray(mid_identity +
                                             self.local_basis.translation_matrix[None, :, :] /
                                             self.J[:, None, None].get())

        # quad matrix
        self.fourier_quads = (self.local_basis.weights[None, None, :] *
                              np.exp(-1j * self.modes[:, None, None] * self.arr[None, :, :]) /
                              self.J[None, :, None].get()) / self.length

        # spectral coefficients
        # self.nyquist_number = 2.0 * self.length // self.dx
        # self.k1 = 2.0 * np.pi / self.length  # fundamental mode
        # # self.wave_numbers = self.k1 * np.arange(1 - self.nyquist_number, self.nyquist_number)
        # self.wave_numbers = self.k1 * np.arange(self.nyquist_number)
        # self.d_wave_numbers = cp.asarray(self.wave_numbers)
        # self.grid_phases = cp.asarray(np.exp(1j * np.tensordot(self.wave_numbers, self.arr, axes=0)))
        #
        # # Spectral matrices
        # self.spectral_transform = self.local_basis.fourier_transform_array(self.mid_points, self.J, self.wave_numbers)
        # self.inverse_transform = self.local_basis.inverse_transform_array(self.mid_points, self.J, self.wave_numbers)

    def create_even_grid(self):
        """ Build global grid """
        # translate to [0, 1]
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        # element left boundaries (including ghost elements)
        xl = np.linspace(self.low, self.high - self.dx, num=self.elements)
        # construct coordinates
        self.arr = np.zeros((self.elements, self.order))
        for i in range(self.elements):
            self.arr[i, :] = xl[i] + self.dx * np.array(nodes_iso)
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])

    def stretch_grid(self):
        # Investigate grid mapping
        # alphas, betas = (np.array([0.64, 0.64, 0.64, 0.64, 0.64]),  # , 0.705, 0.705, 0.705, 0.705]),
        #                  np.array([0.5, 0.5, 0.5, 0.5, 0.5]))
        # alphas, betas = (np.array([0.62, 0.62, 0.62, 0.62]),  # , 0.705, 0.705, 0.705, 0.705]),
        #                  np.array([0.5, 0.5, 0.5, 0.5]))  # , 0.6, 0.6, 0.6, 0.6]))
        alphas, betas = (np.array([1]), np.array([1]))
        # plt.figure()
        # plt.plot(self.arr.flatten(), self.iterate_map_asym(self.arr, alphas[:-2], betas[:-2]).flatten(),
        #          'k', label=r'Iteration 1')
        # plt.plot(self.arr.flatten(), self.iterate_map_asym(self.arr, alphas[:-1], betas[:-1]).flatten(),
        #          'k', label=r'Iteration 2')
        # plt.plot(self.arr.flatten(), self.iterate_map_asym(self.arr, alphas, betas).flatten(),
        #          'r', label=r'Iteration 3')
        # plt.plot(self.arr.flatten(), self.grid_map_asym(self.arr, alpha=0.25, beta=0.5).flatten(),
        #          'b', label=r'$\alpha=0.25, \beta=0.5$')
        # plt.plot(self.arr.flatten(), self.grid_map_asym(self.arr, alpha=0.5, beta=0.25).flatten(),
        #          'g', label=r'$\alpha=0.5, \beta=0.25$')
        # plt.plot(self.arr.flatten(), self.iterate_map(self.arr, orders=[0.5]).flatten(),
        #          'k', label='Iteration 1')
        # plt.plot(self.arr.flatten(), self.iterate_map(self.arr, orders=[0.5, 0.5]).flatten(),
        #          'g', label='Iteration 2')
        # plt.plot(self.arr.flatten(), self.iterate_map(self.arr, orders=[0.5, 0.5, 0.9]).flatten(),
        #          'r', label='Iteration 3')
        # plt.plot(self.arr.flatten(), self.iterate_map(self.arr, orders=[0.5, 0.5, 0.9, 0.9, 0.9, 0.9]).flatten(),
        #          'b', label='Iteration 4')
        # plt.plot(self.arr.flatten(), self.iterate_map(self.arr, orders=[0.25, 0.5, 0.8, 0.8, 0.8]).flatten(),
        #          'k', label='Iteration 5')
        # plt.plot(self.arr.flatten(), self.iterate_map(self.arr, orders=[0.25, 0.5, 0.8, 0.8, 0.8, 0.8]).flatten(),
        #          'k', label='Iteration 6')
        # plt.xlabel('Input points'), plt.ylabel('Output points')
        # plt.grid(True), plt.legend(loc='best'), plt.tight_layout()
        # plt.show()
        # Map points
        # Map lows and highs
        # orders = [0.4, 0.5, 0.8, 0.8, 0.8]  # , 0.8, 0.8]
        # mapped_lows = self.iterate_map(self.arr[:, 0], orders=orders)
        # mapped_highs = self.iterate_map(self.arr[:, -1], orders=orders)
        # alphas, betas = np.array([0.3, 0.55]), np.array([0.35, 0.6])
        mapped_lows = self.iterate_map_asym(self.arr[:, 0], alphas=alphas, betas=betas)
        mapped_highs = self.iterate_map_asym(self.arr[:, -1], alphas=alphas, betas=betas)
        self.dx_grid = mapped_highs - mapped_lows
        # self.dx_grid = self.dx * self.grid_map(self.mid_points)  # mapped_highs - mapped_lows
        # print(self.dx_grid)
        # xl = np.linspace(self.low, self.high - self.dx, num=self.elements)
        # Overwrite coordinate array
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        lows = np.zeros(self.elements + 1)
        lows[0] = self.low
        for i in range(self.elements):
            self.arr[i, :] = lows[i] + self.dx_grid[i] * np.array(nodes_iso)
            lows[i + 1] = self.arr[i, -1]
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])

    def iterate_map(self, points, orders):
        for order in orders:
            points = self.grid_map(points, order)
        return points

    def iterate_map_asym(self, points, alphas, betas):
        for idx, alpha in enumerate(alphas):
            points = self.grid_map_asym(points, alpha, betas[idx])
        return points

    def grid_map_asym(self, points, alpha, beta):
        return (self.low * ((self.high - points) / self.length) ** alpha +
                self.high * ((points - self.low) / self.length) ** beta)

    def grid_map(self, points, order):
        return (self.low * ((self.high - points) / self.length) ** order +
                self.high * ((points - self.low) / self.length) ** order)

    # def fourier_basis(self, function, idx):
    #     """
    #     On GPU, compute Fourier coefficients on the LGL grid of the given grid function
    #     """
    #     return cp.tensordot(function, self.spectral_transform, axes=(idx, [0, 1])) * self.dx / self.length
    #
    # def sum_fourier(self, coefficients, idx):
    #     """
    #     On GPU, re-sum Fourier coefficients up to pre-set cutoff
    #     """
    #     return cp.tensordot(coefficients, self.grid_phases, axes=(idx, [0]))

# # quad fourier transform array
