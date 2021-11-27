import numpy as np
import cupy as cp
import scipy.signal as sig

import matplotlib.pyplot as plt


def dielectric_function(distribution, grid_v, k):
    """
    Compute Vlasov's dielectric function using straight-up quadrature
    """
    k_sq = k ** 2.0
    # Principal-value integral
    # integrand = distribution.grad[None, :, :] / (zeta[:, None, None] - grid_v.device_arr[None, :, :] + 1.0e-2j)
    # integral = cp.tensordot(grid_v.global_quads, integrand, axes=([0, 1], [1, 2]))
    # hilbert = sig.hilbert()
    fourier_transform = grid_v.fourier_basis(function=distribution.grad, idx=[0, 1])
    # sum twice the positive frequency components of fourier series
    hilbert = grid_v.sum_fourier(coefficients=2 * fourier_transform, idx=0)
    # print(distribution.grad.shape)
    # print(fourier_transform.shape)
    # print(hilbert.shape)
    # quit()
    # plt.figure()
    # plt.plot(grid_v.arr.flatten(), cp.imag(hilbert).get().flatten(), 'o--', label='imag f+ fourier')
    # plt.plot(grid_v.arr.flatten(), cp.real(hilbert).get().flatten(), 'o--', label='real f+ fourier')
    # plt.plot(grid_v.arr.flatten(), distribution.grad.get().flatten(), 'o--', label='distribution gradient')
    # plt.xlabel('Velocity'), plt.ylabel('Function')
    # plt.grid(True), plt.legend(loc='best'), plt.tight_layout(), plt.show()
    # hilbert = cp.tensordot()
    dielectric = 1.0 + cp.imag(hilbert) / k_sq * np.pi

    # find zero-crossing


    return dielectric

# def imaginary_part()

    # return hilbert
    #
    # plt.figure()
    # plt.plot(zeta.get(), dielectric.get())
    # plt.show()

