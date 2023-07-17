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

# soln contains the solution in the following order: uL, vL, wL, sL, uN, vN, wN, sN
# u = transverse displacement; v = transverse velocity
# w = longitudinal displacement; s = longitudinal velocity
# L = linear; N = nonlinear

# all the fields are defined on the cell edges x

# Zero Dirichlet boundary conditions are imposed on all fields
# u(0, t) = 0 = u(L, t)
# w(0, t) = 0 = w(L, t)
# v(0, t) = 0 = v(L, t)
# s(0, t) = 0 = s(L, t)

# 230609 TODO LIST:
    # add dispersion into the eqn (kappa value) [DONE i think]
    # change BCs so u=v=0 at edges [life is a struggle]
    # add v into plot so superimposed with u (solid & dashed lines) [DONE]
    # find realistic parameters [see reseaerch document]
    # remove frame png's [DONE]

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

### Input parameters
L  = 0.631                                       # length of domain                
N  = 90                                      # number of grid points
dx = L/N                                      # grid spacing
c2_t = 330                                    # transverse wave speed (squared)
c2_l = 350                                    # longitudinal wave speed (squared)
k  = 1.0                                      # dispersion parameter

t0, tf  = 0, 1                               # initial time, final time
dt, ts  = 0.00001, 0.00005                          # time steps soln and output
tp      = dt*10                               # time step for plotting

### Compute Parameters
Nt  = int(tf/dt)                               # mumber of time steps
npt = int(tp/dt)                               # mumber of time steps to plot
nsv = int(ts/dt)                               # mumber of time steps to save

### Store parameters in a class then output some info
parms = parameters(N = N, L = L, dx = dx, \
                   dt = dt, tf = tf, ts = ts, Nt = Nt, npt = npt, nsv = nsv, skip = 5, \
                   c2_t = c2_t, c2_l = c2_l, k = k, method = flux_wave_eqn)
output_info(parms)

### Initial Conditions with plot: u1, h1, u3, h3
x    = np.linspace(-L/2, L/2, N+1)          # define grids (staggered grid)
# ICs: initial velocity
#soln = np.vstack([0*x, np.exp(-(x**2)/(L/20)**2), \
#                  0*x, np.exp(-(x**2)/(L/20)**2)])
soln = np.vstack([0*x, 0.7*np.exp(-(x**2)/(L/20)**2), 0*x, 0.7*np.exp(-(x**2)/(L/20)**2), \
                  0*x, 0.7*np.exp(-(x**2)/(L/20)**2), 0*x, 0.7*np.exp(-(x**2)/(L/20)**2),])

### Store data to plot later
soln_save = np.zeros((8, N+1, round(tf/ts) + 1))
soln_save[:,:,0] = soln

### Start plotting snapshots
fig, axs = plt.subplots(2, 2, sharex=True)      
plot_soln(x, soln, parms, fig, axs, movie, 0)

### Calculate the solution
soln_save = calculate_soln(x, soln, soln_save, parms, fig, axs, movie)
plt.savefig("final_displacement.png")
plt.show()

### Make animation
if movie:
    merge_to_mp4('frame_%04d.png', movie_name)

## Hovmoller plots of the solution and save
plot_hovmoller(x, soln_save, parms)