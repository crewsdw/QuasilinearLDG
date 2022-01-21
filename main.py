import numpy as np
import cupy as cp
import basis as b
import grid as g
import fluxes as fx
import variables as var
import dielectric
import global_system as global_system
import stepper
import data

import matplotlib.pyplot as plt

# Restart flag
restart = False
# restart_filename = 'bot_2022_jan6_t55_restarted60_twice2'
# save_filename = 'bot_2022_jan6_t55_restarted60_restarted72'
# save_filename = 'bot_2022_jan18_restarted_t50_restarted_t60'
# restart_filename = 'bot_2022_jan18_restarted_t50'
save_filename = 'bot_2022_jan19_resonant'
time = 0

# Type flag
resonant_only = True

# elements and order
length = 5000
elements_v, order = 100, 10
elements_k, order_k = 5, 20

# Bump-on-tail parameters
v0, vt0 = 0, 1
vb, chi = 5, 0.05
vtb = chi ** (1 / 3) * vb

# Set up velocity and wavenumber grids
grid_v = g.VelocityGrid(low=-20, high=30, elements=elements_v, order=order)
if resonant_only:
    grid_k = g.VelocityGrid(low=-20, high=30, elements=elements_v, order=order)
    # grid_k = g.SpectralGrid(low=0.15, high=0.4, elements=elements_k, order=order_k)
else:
    grid_k = g.FiniteSpectralGrid(length=length, modes=length // 2)

# Load data and run from this IC
if restart:
    # Read data
    DataFile = data.Data(folder='bot\\', filename=restart_filename)
    time_data, distribution_data, spectrum_data = DataFile.read_file()
    time = time_data[-1]
    # Set variables
    distribution = var.Scalar(resolution=elements_v, order=order)
    distribution.arr = cp.asarray(distribution_data[-1])
    distribution.reinitialize_tail(grid=grid_v, v0=v0, vt0=vt0, vb=vb, vtb=vtb, chi=chi)
    field = var.EnergySpectrum()
    field.arr = cp.asarray(spectrum_data[-1])
    phase_velocities, growth_rates, previous_guess = dielectric.solve_approximate_dielectric_function(
        distribution=distribution,
        grid_v=grid_v, grid_k=grid_k)
else:
    distribution = var.Scalar(resolution=elements_v, order=order)
    distribution.initialize_particle_distribution(grid=grid_v, v0=v0, vt0=vt0, vb=vb, vtb=vtb, chi=chi)

    # solve dielectric function
    if resonant_only:
        phase_velocities, growth_rates = dielectric.growth_rates_approximate(distribution=distribution,
                                                                             grid_v=grid_v, grid_k=grid_k)
    else:
        phase_velocities, growth_rates, previous_guess = dielectric.solve_approximate_dielectric_function(
            distribution=distribution,
            grid_v=grid_v, grid_k=grid_k)

    # initialize energy spectrum
    pi2 = 2.0 * np.pi
    # field = var.Scalar(resolution=elements_k, order=order)
    field = var.EnergySpectrum()
    if resonant_only:
        field.initialize_spectral_distribution_velocity(grid=grid_k, initial_energy=1e-6 * length / (2.0 * np.pi))
    else:
        field.initialize_spectral_distribution(grid=grid_k, growth_rates=growth_rates * grid_k.arr,
                                               initial_energy=1e-6)  # finite interval
    # field.initialize_spectral_distribution(grid=grid_k, growth_rates=growth_rates * grid_k.arr,
    #                                        initial_energy=1e-5 * length / (2.0 * np.pi))
    #

# Set-up global system
GlobalSystem = global_system.Global(grid=grid_v)

# Get initial dists
initial_dist = distribution.arr.flatten().get()
initial_spec = field.arr.flatten().get()

# Check global system
Flux = fx.DGFlux(resolution=elements_v, order=order)

plt.figure()
if resonant_only:
    plt.plot(grid_k.arr.flatten(), phase_velocities.flatten().get(), 'ro--', label='Real part')
    plt.plot(grid_k.arr.flatten(), growth_rates.flatten().get(), 'go--', label='Imaginary part')
else:
    plt.plot(grid_k.arr.flatten(), phase_velocities.flatten(), 'ro--', label='Real part')
    plt.plot(grid_k.arr.flatten(), growth_rates.flatten(), 'go--', label='Imaginary part')
plt.ylabel(r'Phase velocity $\zeta/v_t$'), plt.xlabel(r'Wavenumber $k\lambda_D$')
plt.legend(loc='best'), plt.grid(True), plt.tight_layout()

# plt.figure()
# plt.plot(grid_k.arr.flatten(), phase_velocities.flatten() * grid_k.arr.flatten(), 'ro--', label='Real part')
# plt.plot(grid_k.arr.flatten(), growth_rates.flatten() * grid_k.arr.flatten(), 'go--', label='Imaginary part')
# plt.plot(grid_k.arr.flatten(), growth_rates / phase_velocities, 'k--', label=r'Ratio $\omega_r/\omega_i$')
# plt.ylabel(r'Frequency $\omega/\omega_p$'), plt.xlabel(r'Wavenumber $k\lambda_D$')
# plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

plt.show()

# Diffusivity
# diffusivity = dielectric.diffusion_coefficient(field_distribution=field,
#                                                grid_v=grid_v, grid_k=grid_k,
#                                                phase_velocity=phase_velocities,
#                                                growth_rates=grid_k.arr * growth_rates)
if resonant_only:
    diffusivity = dielectric.diffusion_coefficient_approximate(field_distribution=field,
                                                               grid_v=grid_v, grid_k=grid_k)
else:
    diffusivity = dielectric.diffusion_coefficient_finite_interval(spectrum=field,
                                                                   grid_v=grid_v, grid_k=grid_k,
                                                                   phase_velocity=phase_velocities,
                                                                   growth_rates=grid_k.arr * growth_rates)
grad, dudt = Flux.semi_discrete_rhs(distribution=distribution, diffusivity=diffusivity, grid=grid_v)

plt.figure()
plt.plot(grid_k.arr.flatten(), field.arr.get().flatten(), 'o--')
plt.grid(True), plt.title('Spectrum'), plt.tight_layout()

plt.figure()
if resonant_only:
    plt.plot(grid_v.arr.flatten(), diffusivity.flatten().get(), 'o--')
else:
    plt.plot(grid_v.arr.flatten(), diffusivity.flatten(), 'o--')
# plt.plot(grid_v.arr.flatten(), diffusivity.flatten().get(), 'o--')
plt.grid(True), plt.title('Diffusivity'), plt.tight_layout()

plt.figure()
plt.semilogy(grid_v.arr.flatten(), distribution.arr.flatten().get(), 'o--')
# plt.xlim([0, 10])
plt.grid(True), plt.title('Initial dist'), plt.tight_layout()

plt.figure()
plt.plot(grid_v.arr.flatten(), grad.flatten().get(), 'o--')
plt.xlim([0, 10])
plt.grid(True), plt.title('Initial turbulent flux'), plt.tight_layout()

plt.figure()
plt.plot(grid_v.arr.flatten(), dudt.flatten().get(), 'o--')
plt.xlim([0, 10])
plt.grid(True), plt.title('Initial dudt'), plt.tight_layout()

plt.show()

# Set up time-stepper and time info
# dt_initial = 0.9 / (2.0 * cp.amax(growth_rates * grid_k.arr))
# dt_initial = 0.9 / (2.0 * cp.amax(growth_rates * grid_k.device_arr))
dt_initial = 0
print('Initial dt is {:0.3e}'.format(dt_initial))
steps = 401  # 501  # 241
mod_idx = 100  # how many per step to view

# Save data
DataFile = data.Data(folder='bot\\', filename=save_filename)
if resonant_only:
    # DataFile.create_file(distribution=distribution.arr.get(),
    #                      spectrum=field.arr.get(), time=time)
    a = 0
else:
    DataFile.create_file(distribution=distribution.arr.get(),
                         spectrum=field.arr.get(), time=time)

# Stepper set-up and loop
TimeStepper = stepper.StepperMidpointMethod(dt=dt_initial, resolution_v=elements_v,
                                            resolution_k=grid_k.arr.shape[0],
                                            order_v=order, order_k=order_k, steps=steps,
                                            initial_t=time, res_flag=resonant_only)
TimeStepper.main_loop(Distribution=distribution, Spectrum=field, GridV=grid_v, GridK=grid_k, GlobalSystem=GlobalSystem)

print('All done')

# Save data
if resonant_only:
    b = 0
else:
    DataFile.save_data(distribution=TimeStepper.saved_arrs[-1, :, :].get(),
                       spectrum=TimeStepper.saved_spectra[-1, :].get(), time=TimeStepper.times[-1].get())

DummyDistribution = var.Scalar(resolution=elements_v, order=order)
DummySpectrum = var.EnergySpectrum()

phase_velocities_arr, growth_rates_arr = (np.zeros_like(TimeStepper.saved_spectra.get()),
                                          np.zeros_like(TimeStepper.saved_spectra.get()))
flux_arr = np.zeros_like(TimeStepper.saved_arrs.get())
diffusivity_arr = np.zeros_like(TimeStepper.saved_arrs.get())

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
# plt.plot(grid_v.arr.flatten(), np.log(initial_dist), 'o--', label=r'$t=0$')
for idx, time in enumerate(TimeStepper.times):
    # Plot velocity
    if idx % mod_idx == 0:
        plt.plot(grid_v.arr.flatten(), np.log(np.abs(TimeStepper.saved_arrs[idx, :, :].flatten().get())), linewidth=3,
                 label=r'$t=${:0.0f}'.format(time.get()))
        # Get growth rates
        # continuum:
        # DummyDistribution.arr = TimeStepper.saved_arrs[idx, :, :]
        # DummySpectrum.arr = TimeStepper.saved_spectra[idx, :, :]
        # finite interval:
        DummyDistribution.arr, DummySpectrum.arr = TimeStepper.saved_arrs[idx, :, :], TimeStepper.saved_spectra[idx, :]

        if resonant_only:
            phase_velocities, growth_rates = dielectric.growth_rates_approximate(distribution=DummyDistribution,
                                                                                 grid_v=grid_v, grid_k=grid_k)
        else:
            phase_velocities, growth_rates, previous_guess = dielectric.solve_approximate_dielectric_function(
                distribution=DummyDistribution,
                grid_v=grid_v, grid_k=grid_k, previous_guess=TimeStepper.previous_guess)
        # print(idx)
        # phase_velocities_arr[idx, :, :] = phase_velocities
        # growth_rates_arr[idx, :, :] = growth_rates
        print(idx)
        if resonant_only:
            phase_velocities_arr[idx, :, :] = phase_velocities.get()
            growth_rates_arr[idx, :, :] = growth_rates.get()
        # finite interval:
        else:
            phase_velocities_arr[idx, :] = phase_velocities
            growth_rates_arr[idx, :] = growth_rates
        if resonant_only:
            diffusivity = dielectric.diffusion_coefficient_approximate(field_distribution=DummySpectrum,
                                                                       grid_v=grid_v, grid_k=grid_k)
        else:
            diffusivity = dielectric.diffusion_coefficient_finite_interval(spectrum=DummySpectrum,
                                                                           grid_v=grid_v, grid_k=grid_k,
                                                                           phase_velocity=phase_velocities,
                                                                           growth_rates=grid_k.arr * growth_rates)
        # diffusivity = dielectric.diffusion_coefficient(field_distribution=DummySpectrum,
        #                                                grid_v=grid_v, grid_k=grid_k,
        #                                                phase_velocity=phase_velocities,
        #                                                growth_rates = grid_k.arr * growth_rates)
        if resonant_only:
            diffusivity_arr[idx, :, :] = diffusivity.get()
        else:
            diffusivity_arr[idx, :, :] = diffusivity

        # Compute turbulent flux
        flux, du_dt = Flux.semi_discrete_rhs(distribution=DummyDistribution, diffusivity=diffusivity, grid=grid_v)
        flux_arr[idx, :, :] = flux.get()
        # diffusivity_arr[idx, :, :] = diffusivity.get()

plt.xlabel(r'Velocity $v/v_t$'), plt.ylabel(r'Distribution function $f(v)$')
plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

# Plot dispersion relations
plt.figure()
for idx, time in enumerate(TimeStepper.times):
    if idx % mod_idx == 0:
        plt.plot(grid_k.arr.flatten(), grid_k.arr.flatten() * growth_rates_arr[idx, :].flatten(), linewidth=3,
                 label=r'$t=${:0.0f}'.format(time.get()))
        # plt.plot(grid_k.arr.flatten(), growth_rates_arr[idx, :].flatten(), 'o--',
        #          label=r'$t=${:0.3f}'.format(time.get()))
plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Growth rate $\omega_i(k)/\omega_p$')
plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

plt.figure()
for idx, time in enumerate(TimeStepper.times):
    if idx % mod_idx == 0:
        plt.plot(grid_k.arr.flatten(), grid_k.arr.flatten() * phase_velocities_arr[idx, :].flatten(), linewidth=3,
                 label=r'$t=${:0.0f}'.format(time.get()))
plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Real frequency $\omega_r(k)/\omega_p$')
plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

# Plot diffusivities
plt.figure()
for idx, time in enumerate(TimeStepper.times):
    if idx % mod_idx == 0:
        plt.plot(grid_v.arr.flatten(), diffusivity_arr[idx, :, :].flatten(), linewidth=3,
                 label=r'$t=${:0.0f}'.format(time.get()))
plt.xlabel(r'Velocity $v/v_t$'), plt.ylabel(r'Diffusivity $D(v)$')
plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

plt.figure()
for idx, time in enumerate(TimeStepper.times):
    if idx % mod_idx == 0:
        plt.plot(grid_v.arr.flatten(), flux_arr[idx, :, :].flatten(), linewidth=3,
                 label=r'$t=${:0.0f}'.format(time.get()))
plt.xlabel(r'Velocity $v/v_t$'), plt.ylabel(r'Turbulent flux $D(v)\partial_v f_0$')
plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

# Plot difference from initial avg_f
plt.figure()
plt.plot(grid_v.arr.flatten(), TimeStepper.saved_arrs[-1, :, :].flatten().get() - initial_dist, linewidth=3)
plt.xlabel(r'Velocity $v/v_t$'), plt.ylabel(r'Final difference from initial distribution, $\Delta f$')
plt.grid(True), plt.tight_layout()

# Plot power spectrum
plt.figure()
# plt.loglog(grid_k.arr.flatten(), initial_spec.flatten(), 'o--', label=r'$t=0$')
# plt.plot(grid_k.arr.flatten(), initial_spec.flatten(), 'o', label=r'$t=0$')
for idx, time in enumerate(TimeStepper.times):
    if idx % mod_idx == 0:
        if resonant_only:
            plt.plot(grid_k.arr.flatten(), TimeStepper.saved_spectra[idx, :].flatten().get(), 'o',
                     label=r'$t=${:0.0f}'.format(time.get()))
        else:
            plt.loglog(grid_k.arr.flatten(), TimeStepper.saved_spectra[idx, :].flatten().get(), 'o--',
                       label=r'$t=${:0.0f}'.format(time.get()))
    # else:
    #     plt.plot(grid_k.arr.flatten(), TimeStepper.saved_spectra[idx, :, :].flatten().get())
plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Field energy spectrum $|E_k|^2$')
plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

plt.show()
