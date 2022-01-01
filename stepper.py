import numpy as np
# import scipy.integrate as spint
import time as timer
import fluxes as fx
import variables as var
import matplotlib.pyplot as plt
import cupy as cp
import dielectric


class StepperMidpointMethod:
    def __init__(self, dt, resolution_v, resolution_k, order_v, order_k, steps):
        self.res_v, self.res_k, self.order = resolution_v, resolution_k, order_v
        self.dt, self.steps = dt, steps
        self.time = 0

        self.SpectrumHalf = var.Scalar(resolution=self.res_k, order=order_k)
        self.Flux = fx.DGFlux(resolution=self.res_v, order=order_v)

        self.saved_arrs = cp.zeros((steps, self.res_v, self.order))
        self.saved_spectra = cp.zeros((steps, self.res_v, self.order))
        # self.saved_spectra = cp.zeros((steps, resolution_k, order_k))
        # self.saved_spectra = cp.zeros((steps, resolution_k))  # finite-interval
        self.times = cp.zeros(steps)
        self.kinetic_energy = cp.zeros(steps)
        self.field_energy = cp.zeros(steps)

    def main_loop(self, Distribution, Spectrum, GridV, GridK, GlobalSystem):
        """ Compute time-steps using the implicit midpoint method """
        for idx in range(self.steps):
            # Adjust time-step
            self.implicit_midpoint(Distribution, Spectrum, GridV, GridK, GlobalSystem)
            self.saved_arrs[idx, :, :] = Distribution.arr
            self.saved_spectra[idx, :, :] = Spectrum.arr
            # self.saved_spectra[idx, :] = Spectrum.arr
            self.kinetic_energy[idx] = Distribution.second_moment(grid=GridV)
            # self.field_energy[idx] = Spectrum.zero_moment()  # finite interval
            # self.field_energy[idx] = Spectrum.zero_moment_continuum(grid=GridK)  # * 2.0 * np.pi / 2000.0
            self.field_energy[idx] = Spectrum.zero_moment_continuum_velocity(grid=GridK)
            self.time += self.dt
            self.times[idx] = self.time
            print('Finished step at time {:0.3e}'.format(self.time.get()))

    def implicit_midpoint(self, distribution, spectrum, grid_v, grid_k, global_system):
        """ Compute the update """
        # Compute growth rates
        # t0 = timer.time()
        # phase_velocities, growth_rates = dielectric.solve_approximate_dielectric_function(distribution=Distribution,
        #                                                                                   grid_v=GridV, grid_k=GridK)
        # t1 = timer.time()
        # Convert growth rate as phase velocity to frequency
        # growth_rates = GridK.device_arr * cp.array(growth_rates)
        phase_velocities, growth_rates = dielectric.growth_rates_approximate(distribution=distribution,
                                                                             grid_v=grid_v, grid_k=grid_k)

        # Adjust time-step
        self.dt = 0.1 / (2.0 * cp.amax(growth_rates))
        # if self.dt > 10:
        #     self.dt = 10.0 * self.dt / self.dt
        # if self.dt > 1:
        self.dt = 0.1 * self.dt / self.dt
        print('This dt is {:0.3e}'.format(self.dt.get()))

        # Take half-step of spectral energy
        # self.SpectrumHalf.arr = Spectrum.arr + 0.5 * self.dt * (2.0 * growth_rates * Spectrum.arr)
        # self.SpectrumHalf.arr = Spectrum.arr * np.exp(2.0 * growth_rates * (self.dt / 2))
        spectrum_rhs = fx.quadratic_basis_product(flux=2.0 * spectrum.arr[:, :, None] * growth_rates[:, None, :],
                                                  basis_arr=grid_k.local_basis.quadratic_source_matrix, axes=[1, 2])
        self.SpectrumHalf.arr = spectrum.arr + 0.5 * self.dt * spectrum_rhs

        # t2 = timer.time()
        # Compute diffusivity at midpoint
        diffusivity = dielectric.diffusion_coefficient_approximate(field_distribution=self.SpectrumHalf,
                                                                   grid_v=grid_v, grid_k=grid_k)
        # diffusivity = dielectric.diffusion_coefficient(field_distribution=self.SpectrumHalf,
        #                                                grid_v=GridV, grid_k=GridK, phase_velocity=phase_velocities,
        #                                                growth_rates=growth_rates.get())
        # diffusivity = dielectric.diffusion_coefficient_finite_interval(spectrum=self.SpectrumHalf,
        #                                                                grid_v=GridV, grid_k=GridK,
        #                                                                phase_velocity=phase_velocities,
        #                                                                growth_rates=growth_rates.get())
        # t3 = timer.time()
        # Construct global system at midpoint
        global_system.build_global_system(diffusivity=diffusivity, flux=self.Flux, grid=grid_v)
        # t4 = timer.time()
        # Update via implicit midpoint rule
        forward_matrix = cp.eye(global_system.size, global_system.size) + 0.5 * self.dt * global_system.matrix
        backward_matrix = cp.linalg.inv(cp.eye(global_system.size, global_system.size) -
                                        0.5 * self.dt * global_system.matrix)
        evolve_matrix = cp.matmul(backward_matrix, forward_matrix)
        # t5 = timer.time()
        # Evolve
        # Spectrum.arr += self.dt * (2.0 * growth_rates * Spectrum.arr)
        # Spectrum.arr = Spectrum.arr * np.exp(2.0 * growth_rates * self.dt)
        spectrum.arr += self.dt * spectrum_rhs
        distribution.arr = cp.matmul(evolve_matrix, distribution.arr.flatten()).reshape(self.res_v, self.order)
        # t6 = timer.time()

        # print(t1-t0)
        # print(t2-t1)
        # print(t3-t2)
        # print(t4-t3)
        # print(t5-t4)
        # print(t6-t5)
        # quit()
