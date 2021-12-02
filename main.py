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
elements_v, elements_k, order = 100, 30, 10

# Set up velocity and wavenumber grids
grid_v = g.VelocityGrid(low=-20, high=20, elements=elements_v, order=order)
grid_k = g.SpectralGrid(low=0.15, high=0.4, elements=elements_k, order=order)

distribution = var.Scalar(resolution=elements_v, order=order)
v0, vt0 = 0, 1
vb, chi = 5, 0.05
vtb = chi ** (1/3) * vb
distribution.initialize_particle_distribution(grid=grid_v, v0=v0, vt0=vt0, vb=vb, vtb=vtb, chi=chi)

# solve dielectric function
phase_velocities, growth_rates = dielectric.solve_approximate_dielectric_function(distribution=distribution,
                                                                      grid_v=grid_v, grid_k=grid_k)

# initialize energy spectrum
field = var.Scalar(resolution=elements_k, order=order)
field.initialize_spectral_distribution(grid=grid_k, growth_rates=growth_rates, initial_energy=1.0e-6)

# Set-up global system
GlobalSystem = global_system.Global(grid=grid_v)

# Set up time-stepper
initial_dist = distribution.arr.flatten().get()
initial_spec = field.arr.flatten().get()

# Time info
dt_initial = 0.45 / (2.0 * cp.amax(growth_rates))
print('Initial dt is {:0.3e}'.format(dt_initial))
steps = 35
TimeStepper = stepper.StepperMidpointMethod(dt=dt_initial, resolution_v=elements_v,
                                            resolution_k=elements_k, order=order, steps=steps)
TimeStepper.main_loop(Distribution=distribution, Spectrum=field, GridV=grid_v, GridK=grid_k, GlobalSystem=GlobalSystem)

print('All done')
DummyDistribution = var.Scalar(resolution=elements_v, order=order)
DummySpectrum = var.Scalar(resolution=elements_k, order=order)

plt.figure()
plt.plot(grid_v.arr.flatten(), initial_dist, 'o--', label=r'$t=0$')
for idx, time in enumerate(TimeStepper.times):
    # Plot velocity
    if idx % 10 == 0:
        plt.plot(grid_v.arr.flatten(), TimeStepper.saved_arrs[idx, :, :].flatten().get(),
                label=r'$t=${:0.3f}'.format(time.get()))
    else:
        plt.plot(grid_v.arr.flatten(), TimeStepper.saved_arrs[idx, :, :].flatten().get())
    # Get growth rates
    DummyDistribution.arr, DummySpectrum.arr = TimeStepper.saved_arrs[idx, :, :], TimeStepper.saved_spectra[idx, :, :]
    phase_velocities, growth_rates = dielectric.solve_approximate_dielectric_function(distribution=DummyDistribution,
                                                                                      grid_v=grid_v, grid_k=grid_k)
    diffusivity = dielectric.diffusion_coefficient(field_distribution=DummySpectrum,
                                                   grid_v=grid_v, grid_k=grid_k, phase_velocity=phase_velocities,
                                                   growth_rates=growth_rates)
plt.xlabel(r'Velocity $v/v_t$'), plt.ylabel(r'Distribution function $f(v)$')
plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

plt.figure()
plt.plot(grid_v.arr.flatten(), TimeStepper.saved_arrs[-1, :, :].flatten().get() - initial_dist, 'o--')
plt.xlabel(r'Velocity $v/v_t$'), plt.ylabel(r'Final difference from initial distribution, $\Delta f$')
plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

plt.figure()
plt.loglog(grid_k.arr.flatten(), initial_spec, 'o--', label=r'$t=0$')
for idx, time in enumerate(TimeStepper.times):
    if idx % 10 == 0:
        plt.plot(grid_k.arr.flatten(), TimeStepper.saved_spectra[idx, :, :].flatten().get(),
                label=r'$t=${:0.3f}'.format(time.get()))
    else:
        plt.plot(grid_k.arr.flatten(), TimeStepper.saved_spectra[idx, :, :].flatten().get())
plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Field spectrum $\mathcal{E}(k)$')
plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

plt.show()
