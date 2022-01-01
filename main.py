import numpy as np
import cupy as cp
import basis as b
import grid as g
import fluxes as fx
import variables as var
import dielectric
import global_system as global_system
import stepper

import matplotlib.pyplot as plt

# elements and order
length = 4000
elements_v, order = 100, 10
elements_k, order_k = 5, 20

# Set up velocity and wavenumber grids
grid_v = g.VelocityGrid(low=-20, high=30, elements=elements_v, order=order)
grid_k = g.VelocityGrid(low=-20, high=30, elements=elements_v, order=order)
# grid_k = g.SpectralGrid(low=0.15, high=0.4, elements=elements_k, order=order_k)
# grid_k = g.FiniteSpectralGrid(length=length, modes=length // 2)

distribution = var.Scalar(resolution=elements_v, order=order)
v0, vt0 = 0, 1
vb, chi = 5, 0.05
vtb = chi ** (1 / 3) * vb
distribution.initialize_particle_distribution(grid=grid_v, v0=v0, vt0=vt0, vb=vb, vtb=vtb, chi=chi)
# print(2.0 * distribution.second_moment(grid=grid_v))
# quit()

# solve dielectric function
# phase_velocities, growth_rates = dielectric.solve_approximate_dielectric_function(distribution=distribution,
#                                                                                   grid_v=grid_v, grid_k=grid_k)
phase_velocities, growth_rates = dielectric.growth_rates_approximate(distribution=distribution,
                                                                     grid_v=grid_v, grid_k=grid_k)

# initialize energy spectrum
pi2 = 2.0 * np.pi
# field = var.Scalar(resolution=elements_k, order=order)
field = var.EnergySpectrum()
# field.initialize_spectral_distribution(grid=grid_k, growth_rates=growth_rates * grid_k.arr,
#                                        initial_energy=1e-5 * length / (2.0 * np.pi))
field.initialize_spectral_distribution_velocity(grid=grid_k, initial_energy=1e-5 * length / (2.0 * np.pi))

# Set-up global system
GlobalSystem = global_system.Global(grid=grid_v)

# Get initial dists
initial_dist = distribution.arr.flatten().get()
initial_spec = field.arr.flatten().get()

# Check global system
Flux = fx.DGFlux(resolution=elements_v, order=order)
# phase_velocities, growth_rates = dielectric.solve_approximate_dielectric_function(distribution=distribution,
#                                                                                   grid_v=grid_v, grid_k=grid_k)

plt.figure()
# plt.plot(grid_k.arr.flatten(), phase_velocities.flatten(), 'o--')
plt.plot(grid_k.arr.flatten(), phase_velocities.flatten().get(), 'o--')
plt.grid(True), plt.title('Real phase velocity'), plt.tight_layout()

plt.figure()
# plt.plot(grid_k.arr.flatten(), growth_rates.flatten() * grid_k.arr.flatten(), 'o--')
plt.plot(grid_k.arr.flatten(), growth_rates.flatten().get(), 'o--')
plt.grid(True), plt.title('Modal growth rates'), plt.tight_layout()

plt.show()

# Diffusivity
# diffusivity = dielectric.diffusion_coefficient(field_distribution=field,
#                                                grid_v=grid_v, grid_k=grid_k,
#                                                phase_velocity=phase_velocities,
#                                                growth_rates=grid_k.arr * growth_rates)
# diffusivity = dielectric.diffusion_coefficient_finite_interval(spectrum=field,
#                                                                grid_v=grid_v, grid_k=grid_k,
#                                                                phase_velocity=phase_velocities,
#                                                                growth_rates=grid_k.arr * growth_rates)
diffusivity = dielectric.diffusion_coefficient_approximate(field_distribution=field,
                                                           grid_v=grid_v, grid_k=grid_k)
grad, dudt = Flux.semi_discrete_rhs(distribution=distribution, diffusivity=diffusivity, grid=grid_v)

plt.figure()
plt.plot(grid_k.arr.flatten(), field.arr.get().flatten(), 'o--')
plt.grid(True), plt.title('Spectrum'), plt.tight_layout()

plt.figure()
# plt.plot(grid_v.arr.flatten(), diffusivity.flatten(), 'o--')
plt.plot(grid_v.arr.flatten(), diffusivity.flatten().get(), 'o--')
plt.grid(True), plt.title('Diffusivity'), plt.tight_layout()

plt.figure()
plt.plot(grid_v.arr.flatten(), dudt.flatten().get(), 'o--')
plt.xlim([2, 7])
plt.grid(True), plt.title('Initial dudt'), plt.tight_layout()

plt.show()

# Set up time-stepper and time info
# dt_initial = 0.9 / (2.0 * cp.amax(growth_rates * grid_k.arr))
dt_initial = 0.9 / (2.0 * cp.amax(growth_rates * grid_k.device_arr))
print('Initial dt is {:0.3e}'.format(dt_initial))
steps = 400  # 501  # 241
TimeStepper = stepper.StepperMidpointMethod(dt=dt_initial, resolution_v=elements_v,
                                            resolution_k=grid_k.arr.shape[0],
                                            order_v=order, order_k=order_k, steps=steps)
TimeStepper.main_loop(Distribution=distribution, Spectrum=field, GridV=grid_v, GridK=grid_k, GlobalSystem=GlobalSystem)

print('All done')
DummyDistribution = var.Scalar(resolution=elements_v, order=order)
DummySpectrum = var.EnergySpectrum()
# DummySpectrum = var.Scalar(resolution=elements_k, order=order)

phase_velocities_arr, growth_rates_arr = (np.zeros_like(TimeStepper.saved_spectra.get()),
                                          np.zeros_like(TimeStepper.saved_spectra.get()))
diffusivity_arr = np.zeros_like(TimeStepper.saved_arrs.get())

# Mod idx
mod_idx = 50

total_energy = TimeStepper.kinetic_energy + TimeStepper.field_energy

print('\nChange in FE is {:0.3e}'.format(TimeStepper.field_energy[-1] - TimeStepper.field_energy[0]))
print('while change of KE is {:0.3e}'.format(TimeStepper.kinetic_energy[-1] - TimeStepper.kinetic_energy[0]))

plt.figure()
plt.semilogy(TimeStepper.times.get(), TimeStepper.kinetic_energy.get(), 'o--', label='Kinetic')
plt.semilogy(TimeStepper.times.get(), TimeStepper.field_energy.get(), 'o--', label='Field')
plt.xlabel('Time $t$'), plt.ylabel(r'Component Energies'), plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

plt.figure()
plt.plot(TimeStepper.times.get(), total_energy.get(), 'o--')
plt.xlabel('Time $t$'), plt.ylabel(r'Total Energy'), plt.grid(True), plt.tight_layout()

plt.figure()
plt.plot(grid_v.arr.flatten(), np.log(initial_dist), 'o--', label=r'$t=0$')
for idx, time in enumerate(TimeStepper.times):
    # Plot velocity
    if idx % mod_idx == 0:
        plt.plot(grid_v.arr.flatten(), np.log(np.abs(TimeStepper.saved_arrs[idx, :, :].flatten().get())), 'o--',
                 label=r'$t=${:0.3f}'.format(time.get()))
        # Get growth rates
        # continuum:
        DummyDistribution.arr = TimeStepper.saved_arrs[idx, :, :]
        DummySpectrum.arr = TimeStepper.saved_spectra[idx, :, :]
        # finite interval:
        # DummyDistribution.arr, DummySpectrum.arr = TimeStepper.saved_arrs[idx,:,:], TimeStepper.saved_spectra[idx, :]

        # phase_velocities, growth_rates = dielectric.solve_approximate_dielectric_function(
        #     distribution=DummyDistribution,
        #     grid_v=grid_v, grid_k=grid_k)
        phase_velocities, growth_rates = dielectric.growth_rates_approximate(distribution=DummyDistribution,
                                                                             grid_v=grid_v, grid_k=grid_k)
        # print(idx)
        # phase_velocities_arr[idx, :, :] = phase_velocities
        # growth_rates_arr[idx, :, :] = growth_rates
        print(idx)
        phase_velocities_arr[idx, :, :] = phase_velocities.get()
        growth_rates_arr[idx, :, :] = growth_rates.get()
        # finite interval:
        # phase_velocities_arr[idx, :] = phase_velocities
        # growth_rates_arr[idx, :] = growth_rates
        diffusivity = dielectric.diffusion_coefficient_approximate(field_distribution=DummySpectrum,
                                                                   grid_v=grid_v, grid_k=grid_k)
        # diffusivity = dielectric.diffusion_coefficient(field_distribution=DummySpectrum,
        #                                                grid_v=grid_v, grid_k=grid_k,
        #                                                phase_velocity=phase_velocities,
        #                                                growth_rates = grid_k.arr * growth_rates)
        # diffusivity = dielectric.diffusion_coefficient_finite_interval(spectrum=DummySpectrum,
        #                                                                grid_v=grid_v, grid_k=grid_k,
        #                                                                phase_velocity=phase_velocities,
        #                                                                growth_rates=grid_k.arr * growth_rates)
        # diffusivity_arr[idx, :, :] = diffusivity
        diffusivity_arr[idx, :, :] = diffusivity.get()
plt.xlabel(r'Velocity $v/v_t$'), plt.ylabel(r'Distribution function $f(v)$')
plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

# Plot dispersion relations
plt.figure()
for idx, time in enumerate(TimeStepper.times):
    if idx % mod_idx == 0:
        # plt.plot(grid_k.arr.flatten(), grid_k.arr.flatten() * growth_rates_arr[idx, :].flatten(), 'o--',
        #          label=r'$t=${:0.3f}'.format(time.get()))
        plt.plot(grid_k.arr.flatten(), growth_rates_arr[idx, :].flatten(), 'o--',
                 label=r'$t=${:0.3f}'.format(time.get()))
plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Growth rate $\omega_i(k)$')
plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

plt.figure()
for idx, time in enumerate(TimeStepper.times):
    if idx % mod_idx == 0:
        plt.plot(grid_k.arr.flatten(), grid_k.arr.flatten() * phase_velocities_arr[idx, :, :].flatten(), 'o--',
                 label=r'$t=${:0.3f}'.format(time.get()))
plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Real frequency $\omega_r(k)$')
plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

# Plot diffusivities
plt.figure()
for idx, time in enumerate(TimeStepper.times):
    if idx % mod_idx == 0:
        plt.plot(grid_v.arr.flatten(), diffusivity_arr[idx, :, :].flatten(), 'o--',
                 label=r'$t=${:0.3f}'.format(time.get()))
plt.xlabel(r'Velocity $v/v_t$'), plt.ylabel(r'Diffusivity $D(v)$')
plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

# Plot difference from initial avg_f
plt.figure()
plt.plot(grid_v.arr.flatten(), TimeStepper.saved_arrs[-1, :, :].flatten().get() - initial_dist, 'o--')
plt.xlabel(r'Velocity $v/v_t$'), plt.ylabel(r'Final difference from initial distribution, $\Delta f$')
plt.grid(True), plt.tight_layout()

# Plot power spectrum
plt.figure()
# plt.loglog(grid_k.arr.flatten(), initial_spec.flatten(), 'o--', label=r'$t=0$')
plt.plot(grid_k.arr.flatten(), initial_spec.flatten(), 'o--', label=r'$t=0$')
for idx, time in enumerate(TimeStepper.times):
    if idx % mod_idx == 0:
        plt.plot(grid_k.arr.flatten(), TimeStepper.saved_spectra[idx, :, :].flatten().get(), 'o--',
                 label=r'$t=${:0.3f}'.format(time.get()))
        # plt.loglog(grid_k.arr.flatten(), TimeStepper.saved_spectra[idx, :, :].flatten().get(), 'o--',
        #            label=r'$t=${:0.3f}'.format(time.get()))
    # else:
    #     plt.plot(grid_k.arr.flatten(), TimeStepper.saved_spectra[idx, :, :].flatten().get())
plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Field spectrum $\mathcal{E}(k)$')
plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

plt.show()
