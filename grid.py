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
        # self.create_even_grid()
        lows = np.array([self.low, 2, 10])
        highs = np.array([2, 10, self.high])
        num_elements = np.array([10, 80, 10])
        self.dx_grid = None
        self.create_triple_grid(lows=lows, highs=highs, elements=num_elements)
        # print(self.arr)
        # quit()

        # stretch / transform elements
        # self.pole_distance = 5
        #
        # self.stretch_grid()
        # self.modes = 2.0 * np.pi * np.arange(1 - int(2 * self.elements), int(2 * self.elements)) / self.length
        # self.modes = 2.0 * np.pi / self.length * cp.arange(int(self.elements))
        self.modes = 2.0 * np.pi / self.length * cp.arange(50)
        self.element_idxs = np.arange(self.elements)

        # jacobian
        self.J = cp.asarray(2.0 / self.dx_grid)
        self.J_host = self.J.get()

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

    def create_triple_grid(self, lows, highs, elements):
        """ Build a three-segment grid, each evenly-spaced """
        # translate to [0, 1]
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        # element left boundaries (including ghost elements)
        dxs = (highs - lows) / elements
        xl0 = np.linspace(lows[0], highs[0] - dxs[0], num=elements[0])
        xl1 = np.linspace(lows[1], highs[1] - dxs[1], num=elements[1])
        xl2 = np.linspace(lows[2], highs[2] - dxs[2], num=elements[2])
        # construct coordinates
        self.arr = np.zeros((elements[0] + elements[1] + elements[2], self.order))
        for i in range(elements[0]):
            self.arr[i, :] = xl0[i] + dxs[0] * nodes_iso
        for i in range(elements[1]):
            self.arr[elements[0] + i, :] = xl1[i] + dxs[1] * nodes_iso
        for i in range(elements[2]):
            self.arr[elements[0] + elements[1] + i, :] = xl2[i] + dxs[2] * nodes_iso
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])
        self.dx_grid = self.device_arr[:, -1] - self.device_arr[:, 0]

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
        # alphas, betas = (np.array([0.64, 0.64, 0.64, 0.64, 0.64, 0.64]),  # , 0.64]),
        #                  np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))  # , 0.5]))
        alphas, betas = (np.array([0.49, 0.49, 0.49]), np.array([0.55, 0.55, 0.55]))  # 0.425, 0.5
        mapped_lows = self.iterate_map_asym(self.arr[:, 0], alphas=alphas, betas=betas)
        mapped_highs = self.iterate_map_asym(self.arr[:, -1], alphas=alphas, betas=betas)
        self.dx_grid = mapped_highs - mapped_lows
        # Overwrite coordinate array
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        lows = np.zeros(self.elements + 1)
        lows[0] = self.low
        for i in range(self.elements):
            self.arr[i, :] = lows[i] + self.dx_grid[i] * np.array(nodes_iso)
            lows[i + 1] = self.arr[i, -1]
        plt.figure()
        plt.plot(self.arr.flatten(), 'o--')
        plt.show()
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


class FiniteSpectralGrid:
    """ Here the wavenumbers are the lattice modes of a finite interval, length L """

    def __init__(self, length, modes):
        self.length = length
        self.modes = modes
        self.fundamental = 2.0 * np.pi / length

        # set-up lattice grid
        self.all_modes = self.fundamental * np.arange(self.modes)
        self.arr = self.all_modes[(0.14 <= self.all_modes) & (self.all_modes <= 0.40)]
        self.device_arr = cp.asarray(self.arr)
        # print()


class SpectralGrid:
    """ In this experiment, the wavenumber grid is an LGL quadrature grid """

    def __init__(self, low, high, elements, order):
        self.low, self.high = low, high
        self.elements, self.order = elements, order
        self.local_basis = b.LGLBasis1D(order=self.order)

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
        self.modes = 2.0 * np.pi / self.length * np.arange(int(self.elements))
        self.element_idxs = np.arange(self.elements)

        # jacobian
        self.J = cp.asarray(2.0 / self.dx_grid)
        self.J_host = self.J.get()

        # global quad weights
        self.global_quads = cp.tensordot(cp.ones(elements),
                                         cp.asarray(self.local_basis.weights), axes=0)
        self.global_quads_host = self.global_quads.get()

        # global translation matrix
        mid_identity = np.tensordot(self.mid_points, np.eye(self.local_basis.order), axes=0)
        self.translation_matrix = cp.asarray(mid_identity +
                                             self.local_basis.translation_matrix[None, :, :] /
                                             self.J[:, None, None].get())

        # quad matrix
        self.fourier_quads = (self.local_basis.weights[None, None, :] *
                              np.exp(-1j * self.modes[:, None, None] * self.arr[None, :, :]) /
                              self.J[None, :, None].get()) / self.length

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
        alphas, betas = (np.array([1]), np.array([1]))
        mapped_lows = self.iterate_map_asym(self.arr[:, 0], alphas=alphas, betas=betas)
        mapped_highs = self.iterate_map_asym(self.arr[:, -1], alphas=alphas, betas=betas)
        self.dx_grid = mapped_highs - mapped_lows
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
