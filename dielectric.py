import numpy as np
import cupy as cp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time as timer


def solve_approximate_dielectric_function(distribution, grid_v, grid_k):
    # Compute p.v. integral via Hilbert transform of distribution
    # distribution.compute_grad(grid=grid_v)
    # distribution.compute_second_grad(grid=grid_v)
    distribution.fourier_transform(grid=grid_v)
    distribution.fourier_grad(grid=grid_v)
    pv_integral = distribution.hilbert_transform_grad(grid=grid_v)

    # initialize arrays
    solutions = np.zeros_like(grid_k.arr.flatten())
    growth_rates = np.zeros_like(solutions)
    approx_plasma = np.zeros_like(solutions)
    om_bohmgross = np.zeros_like(solutions)
    approx_bohmgross = np.zeros_like(solutions)
    guess = 5.6
    # Check out for various grid_k frequencies
    for idx, wave in enumerate(grid_k.arr.flatten()):
        dielectric = 1.0 - pv_integral / (wave ** 2.0)
        if cp.amin(dielectric) > 0:
            continue

        # print('Examining wave {:0.2f}'.format(wave))

        # plt.figure()
        # plt.plot(grid_v.arr.flatten(), dielectric.get().flatten(), 'o--')
        # plt.grid(True), plt.tight_layout(), plt.show()
        def interpolated_dielectric(phase_velocity):
            vidx, velocity = grid_v.get_local_velocity(phase_velocity=
                                                       phase_velocity)
            interpolant_on_point = grid_v.get_interpolant_on_point(velocity=velocity)
            dielectric_on_point = np.tensordot(dielectric[vidx, :].get(), interpolant_on_point, axes=([1], [0]))
            return dielectric_on_point[0]

        def interpolated_dielectric_grad(phase_velocity):
            vidx, velocity = grid_v.get_local_velocity(phase_velocity=
                                                       phase_velocity)
            interpolant_grad_on_point = grid_v.get_interpolant_grad_on_point(velocity=velocity)
            dielectric_grad_on_point = np.tensordot(dielectric[vidx, :].get(), interpolant_grad_on_point,
                                                    axes=([1], [0]))
            return dielectric_grad_on_point[0]

        def grad_on_point(phase_velocity):
            vidx, velocity = grid_v.get_local_velocity(phase_velocity=
                                                       phase_velocity)
            interpolant_grad_on_point = grid_v.get_interpolant_grad_on_point(velocity=velocity)
            return np.tensordot(distribution.arr[vidx, :].get(), interpolant_grad_on_point, axes=([1], [0]))[0]

        # solve it
        solutions[idx] = opt.fsolve(func=interpolated_dielectric, x0=np.array(guess),
                                    fprime=interpolated_dielectric_grad)
        growth_rates[idx] = np.pi * (grad_on_point(phase_velocity=solutions[idx]) /
                                     interpolated_dielectric_grad(phase_velocity=solutions[idx])) / wave ** 2.0
        guess = solutions[idx]

    # t2 = timer.time()
    # print(t1-t0)
    # print(t2-t1)
    # quit()

    # reshape solutions
    solutions = solutions.reshape(grid_k.arr.shape)
    growth_rates = growth_rates.reshape(grid_k.arr.shape)
    growth_rates[(growth_rates < 0) & (np.abs(growth_rates) > 2 * np.amax(growth_rates))] = -2 * np.amax(growth_rates)

    # growth_rates_om = growth_rates * grid_k.arr
    # growth_rates[np.abs(growth_rates_om) < 5.0e-3] = -1.0e-1

    return solutions, growth_rates


def diffusion_coefficient_finite_interval(spectrum, grid_v, grid_k, phase_velocity, growth_rates):
    # Sum terms in the series up to the lattice wavenumber cut-off
    doppler_shifted_f = (phase_velocity[:, None, None] -
                         grid_v.arr[None, :, :]) * grid_k.arr[:, None, None]
    denominator = doppler_shifted_f ** 2.0 + growth_rates[:, None, None] ** 2.0
    # remove terms close to zero
    growth_rates[np.abs(growth_rates) < 0.003] = 0
    plancheral = (2.0 * np.abs(growth_rates[:, None, None]) *
                  spectrum.arr[:, None, None].get() / denominator)
    # sum the terms
    return plancheral.sum(axis=0)


def diffusion_coefficient(field_distribution, grid_v, grid_k, phase_velocity, growth_rates):
    # Compute continuous-spectrum diffusion coefficient
    doppler_shifted_f = (phase_velocity[:, :, None, None] -
                         grid_v.arr[None, None, :, :]) * grid_k.arr[:, :, None, None]
    denominator = doppler_shifted_f ** 2.0 + growth_rates[:, :, None, None] ** 2.0
    # remove terms close to zero
    # growth_rates[np.abs(growth_rates) < 0.01] = 0
    integrand = (2.0 * (np.abs(growth_rates[:, :, None, None])) *
                 field_distribution.arr[:, :, None, None].get() / denominator)

    # diffusion coefficient: naive integration
    result = np.tensordot(grid_k.global_quads_host / grid_k.J_host[:, None], integrand, axes=([0, 1], [0, 1]))

    return result  # / np.pi * 2000.0


def diffusion_coefficient_approximate(field_distribution, grid_v, grid_k):
    # try the gross approximation
    diff = cp.pi * field_distribution.arr / cp.abs(grid_v.device_arr)
    diff[cp.abs(grid_v.device_arr) < 1.0e-10] = 0
    return diff


def growth_rates_approximate(distribution, grid_v, grid_k):
    # experimenting with the gross approximation
    # obtain gradient
    distribution.compute_grad(grid=grid_v)
    #
    growth_rates = cp.zeros_like(grid_k.device_arr)
    solutions = cp.zeros_like(grid_k.device_arr)
    growth_rates[grid_v.device_arr < 0] = -1.0 * (np.pi * 0.5 * (grid_v.device_arr[grid_v.device_arr < 0] ** 2.0) *
                                                  distribution.grad[grid_v.device_arr < 0])
    growth_rates[grid_v.device_arr > 0] = (np.pi * 0.5 * (grid_v.device_arr[grid_v.device_arr > 0] ** 2.0) *
                                           distribution.grad[grid_v.device_arr > 0])
    return solutions, growth_rates

# Bin
# Dielectric:
# vsq
# vsq = 2.3044
# om = np.sqrt(1.0 + 3.0 * vsq * wave ** 2.0)
# om = np.sqrt(wave ** 2.0 * (1.0 + 3 * 1 * wave ** 2.0) / (0.05 + 1.05 * wave ** 2.0))
# om_bohmgross[idx] = om
# print('\n for wave {:0.3e}'.format(wave) + ' the dielectric gradient is:')
# print(wave * interpolated_dielectric_grad(phase_velocity=solutions[idx]))
# print(0.5 * wave ** 2.0)
# print(0.5 * wave ** 2.0 / (om ** 3.0))
# om = 1.0 + 3.0 * (wave ** 2.0)
# approx_plasma[idx] = 0.5 * np.pi * grad_on_point(phase_velocity=1/wave) / (wave ** 2.0)
# approx_bohmgross[idx] = 0.5 * np.pi * grad_on_point(phase_velocity=om/wave) / (wave ** 2.0)
# print(wave)
# print(solutions[idx])
# Diffusion:
# Visualize
# integrand = integrand.reshape((k.shape[0], v.shape[0]))
# K, V = np.meshgrid(k, v, indexing='ij')
# cb = np.linspace(np.amin(integrand), np.amax(integrand), num=100)
# plt.figure()
# plt.contourf(K, V, integrand, cb)
# plt.title('Diffusivity integrand')
# plt.xlabel('Wavenumber'), plt.ylabel('Velocity'), plt.colorbar(), plt.tight_layout()
#
# plt.figure()
# plt.plot(grid_v.arr.flatten(), result.flatten(), 'o--')
# plt.xlabel(r'Velocity $v/v_t$')
# plt.ylabel(r'Diffusivity $D(v)$')
# plt.grid(True), plt.tight_layout()
#
# plt.show()

# Dielectric plots
# plot
# plt.figure()
# plt.plot(grid_k.arr.flatten(), solutions.flatten(), 'ro--', label='Real part')
# plt.plot(grid_k.arr.flatten(), 20 * growth_rates.flatten(), 'go--', label=r'Imaginary part, $\times 20$')
# plt.plot(grid_k.arr.flatten(), 1.0 / grid_k.arr.flatten(), 'o--', label=r'Phase velocity at plasma frequency')
# plt.plot(grid_k.arr.flatten(), om_bohmgross / grid_k.arr.flatten(), 'o--', label='Phase velocity at bohm-gross freq')
# plt.plot(grid_k.arr.flatten(), 20 * approx_plasma / grid_k.arr.flatten(), 'o--', label=r'Appox imag at plasma freq')
# plt.plot(grid_k.arr.flatten(), 20 * approx_bohmgross / grid_k.arr.flatten(), 'o--', label='Approx imag at bohm-gross')
# plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Phase velocity $\zeta/v_t$')
# # plt.xlim([0, 0.4]) #  , plt.ylim([-0.5, 5.5])
# plt.grid(True), plt.legend(loc='best'), plt.tight_layout()
#
# plt.figure()
# plt.plot(grid_k.arr.flatten(), grid_k.arr.flatten() * solutions.flatten(), 'ro--', label='Real part')
# plt.plot(grid_k.arr.flatten(), grid_k.arr.flatten() * growth_rates.flatten(), 'go--',
#          label=r'Imaginary part')
# plt.plot(grid_k.arr.flatten(), np.ones_like(grid_k.arr.flatten()), 'o--', label='Langmuir frequency')
# plt.plot(grid_k.arr.flatten(), om_bohmgross, 'o--', label='Bohm-Gross frequency')
# plt.plot(grid_k.arr.flatten(), approx_plasma, label='Approximate growth rate at plasma frequency')
# plt.plot(grid_k.arr.flatten(), approx_bohmgross, label='Approximate growth rate at bohm-gross frequency')
# plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Frequency $\omega/\omega_p$')
# # plt.xlim([0, 0.4]) #  , plt.ylim([-0.5, 5.5])
# plt.grid(True), plt.legend(loc='best'), plt.tight_layout()
#
# plt.show()
