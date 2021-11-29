import numpy as np
import cupy as cp
import basis as b
import grid as g
import fluxes as fx
import variables as var
import dielectric

import matplotlib.pyplot as plt

# elements and order
elements_v, elements_k, order = 50, 10, 25

# Set up velocity and wavenumber grids
grid_v = g.VelocityGrid(low=-15, high=15, elements=elements_v, order=order)
grid_k = g.SpectralGrid(low=0.15, high=0.4, elements=elements_k, order=order)

distribution = var.Scalar(resolution=elements_v, order=order)
v0, vt0 = 0, 1
vb, chi = 5, 0.05
vtb = chi ** (1/3) * vb
distribution.initialize_particle_distribution(grid=grid_v, v0=v0, vt0=vt0, vb=vb, vtb=vtb, chi=chi)

# distribution.fourier_transform(grid=grid_v)
# distribution.fourier_grad(grid=grid_v)
# a = distribution.hilbert_transform_grad(grid=grid_v)

# distribution.cauchy_transform_grad(grid=grid_v)

# solve dielectric function
phase_velocities, growth_rates = dielectric.solve_approximate_dielectric_function(distribution=distribution,
                                                                      grid_v=grid_v, grid_k=grid_k)

# initialize energy spectrum
field = var.Scalar(resolution=elements_k, order=order)
field.initialize_spectral_distribution(grid=grid_k, growth_rates=growth_rates, initial_energy=1.0e-6)

dielectric.diffusion_coefficient(field_distribution=field, grid_v=grid_v, grid_k=grid_k,
                                 phase_velocity=phase_velocities, growth_rates=growth_rates)
quit()

# quit()

# plt.figure()
# plt.plot(grid_v.arr.flatten(), distribution.grad.flatten().get(), 'o--')
# plt.show()

# check out dielectric function
# zeta = cp.linspace(2, 5, num=100)
k = np.linspace(0.05, 0.6, num=200)
epsilon = 1.0 - a.get()[None, :, :] / (k[:, None, None] ** 2.0)
# epsilon = np.zeros((200, 50, 8))
# for i in range(200):
#     epsilon[i, :, :] = 1.0 - a.get() / (k[i] ** 2.0)
# dielectric.dielectric_function(distribution=distribution, grid_v=grid_v, k=k[i]).get()

K, Z = np.meshgrid(k, grid_v.arr.flatten(), indexing='ij')
cb = np.linspace(-1, 1, num=100)

plt.figure()
plt.contour(K, Z, epsilon.reshape(200, Z.shape[1]), 0)
plt.xlabel(r'Wavenumber $k$'), plt.ylabel(r'Phase velocity $\zeta_r$')
plt.grid(True), plt.tight_layout()
# plt.contourf(K, Z, epsilon.reshape(200, Z.shape[1]), cb, extend='both')
# plt.plot(zeta.get(), epsilon.get(), 'o--')
plt.show()
