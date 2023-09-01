#!/usr/bin/env python
#
######################################################################
#                                                                    #
#  nonlinear_wave_eqn.py                                             #
#                                                                    #
# Written June 8, 2023 by Francis J. Poulin and Sarah T. Moser       #
#                                                                    # 
######################################################################
#
# Methods to run this script:
#  1) from the terminal using
#     $ python3 nonlinear_wave_eqn.py
#
#  2) after starting python3, use
#  >>> import nonlinear_wave_eqn

# soln contains the solution in the following order: uL, vL, wL, sL, uN, vN, wN, sN, uT, vT, wT, sT, pT, vpT
# u = transverse displacement; v = transverse velocity
# w = longitudinal displacement; s = longitudinal velocity
# p = varphi displacement; vp = varphi velocity (Timoshenko model)
# L = linear; N = nonlinear; T = Timoshenko model

# all the fields are defined on the cell edges x

# Zero Dirichlet boundary conditions are imposed on all fields
# u(0, t) = 0 = u(L, t)
# w(0, t) = 0 = w(L, t)
# v(0, t) = 0 = v(L, t)
# s(0, t) = 0 = s(L, t)
# p(0, t) = 0 = p(L, t)

### import standard libraries
import numpy as np                             # numerical library
import matplotlib.pyplot as plt                # plotting library

### import personal libraries
from library import parameters                 # class to store parameters
from library import merge_to_mp4               # to make animation
from library import flux_wave_eqn              # flux for the PDEs to integrate
from library import plot_soln                  # plot snapshots
from library import plot_hovmoller             # plot hovmoller
from library import output_info                # output some info
from library import calculate_soln             # integrate the PDE to find soln

# options to make the movie
movie = True                                   # switch to make an animation
movie_name = 'wave_eqn_movie.mp4'

# save data
outfile = "soln_data.npy"

### Input parameters
L    = 0.961                                   # length of domain                
N    = 50                                     # number of grid points
dx   = L/N                                     # grid spacing
c2_t = 1.13e5                                  # transverse wave speed (squared)
c2_l = 2.55e7                                  # longitudinal wave speed (squared)
k    = 0.95                                    # Timoshenko shear parameter
C1   = 1.44e7                                  # A/I parameter
C2   = 9.68e6                                  # Gk/rho parameter

t0, tf  = 0, 5e-6                              # initial time, final time
dt, ts  = 1e-11, 5e-7                          # time steps soln and output
m       = 100000                               # multiplication factor for tp and movie
tp      = dt*m                                 # time step for plotting

### Compute Parameters
Nt  = int(tf/dt)                               # mumber of time steps
npt = int(tp/dt)                               # mumber of time steps to plot
nsv = int(ts/dt)                               # mumber of time steps to save

### Store parameters in a class then output some info
parms = parameters(N = N, L = L, dx = dx, \
                   dt = dt, tf = tf, ts = ts, m = m, Nt = Nt, npt = npt, nsv = nsv, skip = 5, \
                   c2_t = c2_t, c2_l = c2_l, k = k, C1 = C1, C2 = C2, method = flux_wave_eqn)
output_info(parms)

### Initial Conditions with plot: uL, vL, wL, sL, uN, vN, wN, sN, uT, vT, wT, sT, pT, vpT
x    = np.linspace(-L/2, L/2, N+1)          # define grids (staggered grid)
xs   = np.linspace(-L/2 + dx/2, L/2 - dx/2, N)     # staggered grid
# ICs: initial velocity
soln = np.vstack([0*x, 0.7*np.exp(-(x**2)/(L/20)**2), 0*x, 0.7*np.exp(-(x**2)/(L/20)**2), \
                  0*x, 0.7*np.exp(-(x**2)/(L/20)**2), 0*x, 0.7*np.exp(-(x**2)/(L/20)**2), \
                  0*x, 0.7*np.exp(-(x**2)/(L/20)**2), 0*x, 0.7*np.exp(-(x**2)/(L/20)**2), \
                  np.hstack([0*xs, 0]), np.hstack([0*xs, 0]) ])

### Store data to plot later
soln_save = np.zeros((14, N+1, round(tf/ts) + 1))
soln_save[:,:,0] = soln

### Start plotting snapshots
fig, axs = plt.subplots(2, 3, sharex=True)      
plot_soln(x, xs, soln, parms, fig, axs, movie, 0)

### Calculate the solution
soln_save = calculate_soln(x, xs, soln, soln_save, parms, fig, axs, movie)
plt.savefig("final_displacement.png")
plt.show()

### Save the data into a file
with open(outfile, "wb") as f:
    np.save(f, soln_save)

### Make animation
if movie:
    merge_to_mp4('frame_%04d.png', movie_name)

## Hovmoller plots of the solution and save
plot_hovmoller(x, soln_save, parms)