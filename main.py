import numpy as np
import cupy as cp
import basis as b
import grid as g
import fluxes as fx
import variables as var
import dielectric

import matplotlib.pyplot as plt

# elements and order
elements_v, elements_k, order = 50, 50, 8

# Set up velocity and wavenumber grids
grid_v = g.VelocityGrid(low=-15, high=15, elements=elements_v, order=order)
grid_k = g.SpectralGrid(low=0, high=3, elements=elements_k, order=order)

distribution = var.Scalar(resolution=elements_v, order=order)
v0, vt0 = 0, 1
vb, chi = 5, 0.05
vtb = chi ** (1/3) * vb
distribution.initialize_particle_distribution(grid=grid_v, v0=v0, vt0=vt0, vb=vb, vtb=vtb, chi=chi)

# plt.figure()
# plt.plot(grid_v.arr.flatten(), distribution.grad.flatten().get(), 'o--')
# plt.show()

# check out dielectric function
# zeta = cp.linspace(2, 5, num=100)
k = np.linspace(0.05, 1, num=200)
epsilon = np.zeros((200, 50, 8))
for i in range(200):
    epsilon[i, :, :] = dielectric.dielectric_function(distribution=distribution, grid_v=grid_v, k=k[i]).get()

K, Z = np.meshgrid(k, grid_v.arr.flatten(), indexing='ij')
cb = np.linspace(-1, 1, num=100)

plt.figure()
# plt.contour(K, Z, epsilon.reshape(200, Z.shape[1]), 0)
plt.contourf(K, Z, epsilon.reshape(200, Z.shape[1]), cb, extend='both')
# plt.plot(zeta.get(), epsilon.get(), 'o--')
plt.show()
