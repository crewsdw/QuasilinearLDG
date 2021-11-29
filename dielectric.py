import numpy as np
import cupy as cp
import scipy.optimize as opt
import matplotlib.pyplot as plt


def solve_approximate_dielectric_function(distribution, grid_v, grid_k):
    # Compute p.v. integral via Hilbert transform of distribution
    distribution.compute_grad(grid=grid_v)
    # distribution.compute_second_grad(grid=grid_v)
    distribution.fourier_grad(grid=grid_v)
    pv_integral = distribution.hilbert_transform_grad(grid=grid_v)

    # initialize arrays
    solutions = np.zeros_like(grid_k.arr.flatten())
    growth_rates = np.zeros_like(solutions)
    guesses = np.linspace(5, 3, num=solutions.shape[0])
    # print(guesses)
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
        solutions[idx] = opt.fsolve(func=interpolated_dielectric, x0=np.array([guesses[idx]]))
        growth_rates[idx] = np.pi * (grad_on_point(phase_velocity=solutions[idx]) /
                                     interpolated_dielectric_grad(phase_velocity=solutions[idx])) / (wave ** 2.0)

    # reshape solutions
    solutions = solutions.reshape(grid_k.arr.shape)
    growth_rates = growth_rates.reshape(grid_k.arr.shape)

    # plot
    plt.figure()
    plt.plot(grid_k.arr.flatten(), solutions.flatten(), 'ro--', label='Real part')
    plt.plot(grid_k.arr.flatten(), 20 * growth_rates.flatten(), 'go--', label=r'Imaginary part, $\times 20$')
    plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Phase velocity $\zeta/v_t$')
    plt.xlim([0, 0.4]), plt.ylim([-0.5, 5.5])
    plt.grid(True), plt.legend(loc='best'), plt.tight_layout(), plt.show()

    return solutions, growth_rates


def diffusion_coefficient(field_distribution, grid_v, grid_k, phase_velocity, growth_rates):
    k = grid_k.arr.flatten()
    v = grid_v.arr.flatten()
    # Compute the regularized integrand
    doppler_shifted_f = (phase_velocity[:, :, None, None] - grid_v.arr[None, None, :, :]) * grid_k.arr[:, :, None, None]
    denominator = doppler_shifted_f ** 2.0 + growth_rates[:, :, None, None] ** 2.0
    integrand = (2.0 * np.abs(growth_rates[:, :, None, None]) *
                 field_distribution.arr[:, :, None, None].get() / denominator)

    # non-resonant diffusion coefficient: naive integration
    result = np.tensordot(grid_k.global_quads.get(), integrand, axes=([0, 1], [0, 1]))

    # Visualize
    # plt.figure()
    # plt.plot(grid_k.arr.flatten(), integrand[:, :, 30, 5].flatten(), 'o--')

    integrand = integrand.reshape((k.shape[0], v.shape[0]))
    K, V = np.meshgrid(k, v, indexing='ij')
    cb = np.linspace(np.amin(integrand), np.amax(integrand), num=100)
    plt.figure()
    plt.contourf(K, V, integrand, cb)
    plt.title('Diffusivity integrand')
    plt.xlabel('Wavenumber'), plt.ylabel('Velocity'), plt.colorbar(), plt.tight_layout()

    plt.figure()
    plt.plot(grid_v.arr.flatten(), result.flatten(), 'o--')
    plt.xlabel(r'Velocity $v/v_t$')
    plt.ylabel(r'Diffusivity $D(v)$')
    plt.grid(True), plt.tight_layout()
    plt.show()

    plt.show()

    return
