import numpy as np
# import scipy.integrate as spint
import time as timer
import fluxes as fx
import variables as var
import matplotlib.pyplot as plt
import cupy as cp
import dielectric


class StepperMidpointMethod:
    def __init__(self, dt, resolution_v, resolution_k, order, steps):
        self.res_v, self.res_k, self.order = resolution_v, resolution_k, order
        self.dt, self.steps = dt, steps
        self.time = 0

        self.SpectrumHalf = var.Scalar(resolution=self.res_k, order=self.order)
        self.Flux = fx.DGFlux(resolution=self.res_v, order=order)

        self.saved_arrs = cp.zeros((steps, self.res_v, self.order))
        self.saved_spectra = cp.zeros((steps, self.res_k, self.order))
        self.times = cp.zeros(steps)

    def main_loop(self, Distribution, Spectrum, GridV, GridK, GlobalSystem):
        """ Compute time-steps using the implicit midpoint method """
        for idx in range(self.steps):
            # Adjust time-step
            self.implicit_midpoint(Distribution, Spectrum, GridV, GridK, GlobalSystem)
            self.saved_arrs[idx, :, :] = Distribution.arr
            self.saved_spectra[idx, :, :] = Spectrum.arr
            self.time += self.dt
            self.times[idx] = self.time
            print('Finished step at time {:0.3e}'.format(self.time.get()))

    def implicit_midpoint(self, Distribution, Spectrum, GridV, GridK, GlobalSystem):
        """ Compute the update """
        # Compute growth rates
        phase_velocities, growth_rates = dielectric.solve_approximate_dielectric_function(distribution=Distribution,
                                                                                          grid_v=GridV, grid_k=GridK)

        growth_rates = cp.array(growth_rates)

        # Adjust time-step
        self.dt = 0.45 / (2.0 * cp.amax(growth_rates))
        print('This dt is {:0.3e}'.format(self.dt.get()))

        # Take half-step of spectral energy
        self.SpectrumHalf.arr = Spectrum.arr + 0.5 * self.dt * (2.0 * growth_rates * Spectrum.arr)

        # Compute diffusivity at midpoint
        diffusivity = dielectric.diffusion_coefficient(field_distribution=self.SpectrumHalf,
                                                       grid_v=GridV, grid_k=GridK, phase_velocity=phase_velocities,
                                                       growth_rates=growth_rates.get())

        # Construct global system at midpoint
        GlobalSystem.build_global_system(diffusivity=diffusivity, flux=self.Flux, grid=GridV)

        # Update via implicit midpoint rule
        forward_matrix = cp.eye(GlobalSystem.size, GlobalSystem.size) + 0.5 * self.dt * GlobalSystem.matrix
        backward_matrix = cp.linalg.inv(cp.eye(GlobalSystem.size, GlobalSystem.size) -
                                        0.5 * self.dt * GlobalSystem.matrix)
        evolve_matrix = cp.matmul(backward_matrix, forward_matrix)

        # Evolve
        Spectrum.arr += self.dt * (2.0 * growth_rates * Spectrum.arr)
        Distribution.arr = cp.matmul(evolve_matrix, Distribution.arr.flatten()).reshape(self.res_v, self.order)

