import numpy as np
import cupy as cp
import numpy.polynomial as poly
import scipy.special as sp

# Legendre-Gauss-Lobatto nodes and quadrature weights dictionaries
lgl_nodes = {
    1: [0],
    2: [-1, 1],
    3: [-1, 0, 1],
    4: [-1, -np.sqrt(1 / 5), np.sqrt(1 / 5), 1],
    5: [-1, -np.sqrt(3 / 7), 0, np.sqrt(3 / 7), 1],
    6: [-1, -np.sqrt(1 / 3 + 2 * np.sqrt(7) / 21), -np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21),
        np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21), np.sqrt(1 / 3 + 2 * np.sqrt(7) / 21), 1],
    7: [-1, -0.830223896278566929872, -0.468848793470714213803772,
        0, 0.468848793470714213803772, 0.830223896278566929872, 1],
    8: [-1, -0.8717401485096066153375, -0.5917001814331423021445,
        -0.2092992179024788687687, 0.2092992179024788687687,
        0.5917001814331423021445, 0.8717401485096066153375, 1],
    9: [-1, -0.8997579954114601573124, -0.6771862795107377534459,
        -0.3631174638261781587108, 0, 0.3631174638261781587108,
        0.6771862795107377534459, 0.8997579954114601573124, 1],
    10: [-1, -0.9195339081664588138289, -0.7387738651055050750031,
         -0.4779249498104444956612, -0.1652789576663870246262,
         0.1652789576663870246262, 0.4779249498104444956612,
         0.7387738651055050750031, 0.9195339081664588138289, 1]
}

lgl_weights = {
    1: [2],
    2: [1, 1],
    3: [1 / 3, 4 / 3, 1 / 3],
    4: [1 / 6, 5 / 6, 5 / 6, 1 / 6],
    5: [1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10],
    6: [1 / 15, (14 - np.sqrt(7)) / 30, (14 + np.sqrt(7)) / 30,
        (14 + np.sqrt(7)) / 30, (14 - np.sqrt(7)) / 30, 1 / 15],
    7: [0.04761904761904761904762, 0.2768260473615659480107,
        0.4317453812098626234179, 0.487619047619047619048,
        0.4317453812098626234179, 0.2768260473615659480107,
        0.04761904761904761904762],
    8: [0.03571428571428571428571, 0.210704227143506039383,
        0.3411226924835043647642, 0.4124587946587038815671,
        0.4124587946587038815671, 0.3411226924835043647642,
        0.210704227143506039383, 0.03571428571428571428571],
    9: [0.02777777777777777777778, 0.1654953615608055250463,
        0.2745387125001617352807, 0.3464285109730463451151,
        0.3715192743764172335601, 0.3464285109730463451151,
        0.2745387125001617352807, 0.1654953615608055250463,
        0.02777777777777777777778],
    10: [0.02222222222222222222222, 0.1333059908510701111262,
         0.2248893420631264521195, 0.2920426836796837578756,
         0.3275397611838974566565, 0.3275397611838974566565,
         0.292042683679683757876, 0.224889342063126452119,
         0.133305990851070111126, 0.02222222222222222222222]
}


class LGLBasis1D:
    """
    Class containing basis-related methods and properties for Lobatto-Gauss-Legendre points
    """

    def __init__(self, order):
        # parameters
        self.order = int(order)
        self.nodes, self.weights = (np.array(lgl_nodes.get(self.order, "nothing")),
                                    np.array(lgl_weights.get(self.order, "nothing")))
        self.device_weights = np.asarray(self.weights)

        # vandermonde matrix and inverse
        self.eigenvalues = self.set_eigenvalues()
        self.vandermonde = self.set_vandermonde()
        self.inv_vandermonde = self.set_inv_vandermonde()

        # DG matrices
        self.mass, self.inv_mass = None, None
        self.internal, self.numerical = None, None

        # Set matrices
        self.set_mass_matrix(), self.set_inv_mass_matrix()
        self.set_internal_flux_matrix()
        self.set_numerical_flux_matrix()

        # Compute translation matrix
        self.translation_matrix = None
        self.set_translation_matrix()

    def set_eigenvalues(self):
        evs = np.array([(2.0 * s + 1) / 2.0 for s in range(self.order - 1)])
        return np.append(evs, (self.order - 1) / 2.0)

    def set_vandermonde(self):
        return np.array([[sp.legendre(s)(self.nodes[j])
                          for j in range(self.order)]
                         for s in range(self.order)])

    def set_inv_vandermonde(self):
        return np.array([[self.weights[j] * self.eigenvalues[s] * sp.legendre(s)(self.nodes[j])
                          for j in range(self.order)]
                         for s in range(self.order)])

    def set_mass_matrix(self):
        # Diagonal part
        approx_mass = np.diag(self.weights)

        # Off-diagonal part
        p = sp.legendre(self.order - 1)
        v = np.multiply(self.weights, p(self.nodes))
        a = -self.order * (self.order - 1) / (2.0 * (2.0 * self.order - 1))
        # calculate mass matrix
        self.mass = approx_mass + a * np.outer(v, v)

    def set_inv_mass_matrix(self):
        # Diagonal part
        approx_inv = np.diag(np.divide(1.0, self.weights))

        # Off-diagonal part
        p = sp.legendre(self.order - 1)
        v = p(self.nodes)
        b = self.order / 2
        # calculate inverse mass matrix
        self.inv_mass = approx_inv + b * np.outer(v, v)

    def set_internal_flux_matrix(self):
        # Compute internal flux array
        up = np.zeros((self.order, self.order))
        for i in range(self.order):
            for j in range(self.order):
                up[i, j] = self.weights[j] * sum(
                    (2 * s + 1) / 2 * sp.legendre(s)(self.nodes[i]) *
                    sp.legendre(s).deriv()(self.nodes[j]) for s in range(self.order))

        # Clear machine errors
        up[np.abs(up) < 1.0e-10] = 0

        self.internal = cp.asarray(up)

    def set_numerical_flux_matrix(self):
        self.numerical = cp.asarray(self.inv_mass[:, np.array([0, -1])])
        # print(self.numerical)
        # quit()

    def set_translation_matrix(self):
        """ Create the translation matrix for velocity DG method """
        local_order = self.order
        gl_nodes, gl_weights = poly.legendre.leggauss(local_order)
        # Evaluate Legendre polynomials at finer grid
        ps = np.array([sp.legendre(s)(gl_nodes) for s in range(self.order)])
        # Interpolation polynomials at fine points
        ell = np.tensordot(self.inv_vandermonde, ps, axes=([0], [0]))
        # Compute the matrix elements
        translation_mass = np.array([[
            sum(gl_weights[s] * gl_nodes[s] * ell[i, s] * ell[j, s] for s in range(local_order))
            for j in range(self.order)]
            for i in range(self.order)])
        # Multiply by inverse mass matrix
        self.translation_matrix = np.matmul(self.inv_mass, translation_mass) + 0j

    def fourier_transform_array(self, midpoints, J, wave_numbers):
        """
        Grid-dependent spectral coefficient matrix
        Needs grid quantities: Jacobian, wave numbers, nyquist number
        """
        # Check sign of wave-numbers (see below)
        signs = np.sign(wave_numbers)
        signs[np.where(wave_numbers == 0)] = 1.0

        # Fourier-transformed modal basis ( (-1)^s accounts for scipy's failure to have negative spherical j argument )
        p_tilde = np.array([(signs ** s) * np.exp(-1j * np.pi / 2.0 * s) *
                            sp.spherical_jn(s, np.absolute(wave_numbers) / J) for s in range(self.order)])

        # Multiply by inverse Vandermonde transpose for fourier-transformed nodal basis
        ell_tilde = np.matmul(self.inv_vandermonde.T, p_tilde)

        # Outer product with phase factors
        phase = np.exp(-1j * np.tensordot(midpoints, wave_numbers, axes=0))
        transform_array = np.multiply(phase[:, :, None], ell_tilde.T)

        # Put in order (resolution, nodes, modes)
        transform_array = np.transpose(transform_array, (0, 2, 1))

        # Return as cupy array
        return cp.asarray(transform_array)

    def inverse_transform_array(self, midpoints, J, wave_numbers):
        """
        Grid-dependent spectral coefficient matrix
        Experimental inverse-transform matrix
        """
        # Check sign (see below)
        signs = np.sign(wave_numbers)
        signs[np.where(wave_numbers == 0)] = 1.0
        # wave_numbers[np.where(wave_numbers == 0)] = 1.0

        # Inverse transform array
        # p_tilde = np.array([(signs ** s) * np.exp(1j * np.pi / 2.0 * s) *
        #                     (sp.spherical_jn(s, np.absolute(wave_numbers) / J)) for s in range(self.order)])
        # Multiply by Vandermonde matrix
        # next_step = np.matmul(self.vandermonde.T, p_tilde)
        vandermonde_contraction = np.array(sum(self.vandermonde[i, :] for i in range(self.order)))
        spherical_summation = np.array(sum((signs ** s) * ((-1j) ** s) *  # np.exp(-1j * np.pi / 2.0 * s) *
                                           (sp.spherical_jn(s, np.absolute(wave_numbers) / J)) for s in
                                           range(self.order)))
        # print(vandermonde_contraction.shape)
        # print(spherical_contraction.shape)

        # Outer product with phase factors
        phase = np.exp(1j * np.tensordot(midpoints, wave_numbers, axes=0))
        next_step = np.divide(phase, spherical_summation[None, :])
        inverse_transform = np.transpose(np.tensordot(next_step, vandermonde_contraction, axes=0),
                                         axes=(0, 2, 1))
        # print(inverse_transform.shape)
        # inverse_transform = np.multiply(phase[:, None, :], next_step[None, :, :])

        return cp.asarray(inverse_transform)
