import numpy as np
import cupy as cp
import basis as b


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

        # jacobian
        self.J = 2.0 / self.dx

        # global quad weights
        self.global_quads = cp.tensordot(cp.ones(elements),
                                         cp.asarray(self.local_basis.weights), axes=0)

        # arrays
        self.arr, self.device_arr = None, None
        self.mid_points = None
        self.create_grid()

        # quad fourier transform array
        # spectral coefficients
        self.nyquist_number = 2.0 * self.length // self.dx
        self.k1 = 2.0 * np.pi / self.length  # fundamental mode
        # self.wave_numbers = self.k1 * np.arange(1 - self.nyquist_number, self.nyquist_number)
        self.wave_numbers = self.k1 * np.arange(self.nyquist_number)
        self.d_wave_numbers = cp.asarray(self.wave_numbers)
        self.grid_phases = cp.asarray(np.exp(1j * np.tensordot(self.wave_numbers, self.arr, axes=0)))

        # Spectral matrices
        self.spectral_transform = self.local_basis.fourier_transform_array(self.mid_points, self.J, self.wave_numbers)
        self.inverse_transform = self.local_basis.inverse_transform_array(self.mid_points, self.J, self.wave_numbers)

    def create_grid(self):
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

    def fourier_basis(self, function, idx):
        """
        On GPU, compute Fourier coefficients on the LGL grid of the given grid function
        """
        return cp.tensordot(function, self.spectral_transform, axes=(idx, [0, 1])) * self.dx / self.length

    def sum_fourier(self, coefficients, idx):
        """
        On GPU, re-sum Fourier coefficients up to pre-set cutoff
        """
        return cp.tensordot(coefficients, self.grid_phases, axes=(idx, [0]))


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

        # jacobian
        self.J = 2.0 / self.dx

        # global quad weights
        self.global_quads = cp.tensordot(cp.ones(elements),
                                         cp.asarray(self.local_basis.weights), axes=0)

        # arrays
        self.arr, self.device_arr = None, None
        self.mid_points = None
        self.create_grid()

        # quad fourier transform array
        # spectral coefficients
        self.nyquist_number = 2.0 * self.length // self.dx
        self.k1 = 2.0 * np.pi / self.length  # fundamental mode
        self.wave_numbers = self.k1 * np.arange(1 - self.nyquist_number, self.nyquist_number)
        self.d_wave_numbers = cp.asarray(self.wave_numbers)
        self.grid_phases = cp.asarray(np.exp(1j * np.tensordot(self.wave_numbers, self.arr[1:-1, :], axes=0)))

        # Spectral matrices
        self.spectral_transform = self.local_basis.fourier_transform_array(self.mid_points, self.J, self.wave_numbers)
        self.inverse_transform = self.local_basis.inverse_transform_array(self.mid_points, self.J, self.wave_numbers)

    def create_grid(self):
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

    def fourier_basis(self, function, idx):
        """
        On GPU, compute Fourier coefficients on the LGL grid of the given grid function
        """
        return cp.tensordot(function, self.spectral_transform, axes=(idx, [0, 1])) * self.dx / self.length

    def sum_fourier(self, coefficients, idx):
        """
        On GPU, re-sum Fourier coefficients up to pre-set cutoff
        """
        return cp.tensordot(coefficients, self.grid_phases, axes=(idx, [0]))
