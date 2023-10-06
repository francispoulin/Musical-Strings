#!/usr/bin/env python
#
######################################################################
#                                                                    #
#  linear_wave_eqn.py                                                #
#                                                                    #
# Written Sep 29, 2023 by Francis J. Poulin and Sarah T. Moser       #
#                                                                    # 
######################################################################
#
# Methods to run this script:
#  1) from the terminal using
#     $ python3 linear_wave_eqn.py
#
#  2) after starting python3, use
#  >>> import linear_wave_eqn

# soln contains the solution in the following order: u, v
# u = transverse displacement; v = transverse velocity

# all the fields are defined on the cell edges x

# Zero Dirichlet boundary conditions are imposed on all fields
# u(0, t) = 0 = u(L, t)
# v(0, t) = 0 = v(L, t)

### import standard libraries
import numpy as np                             # numerical library
import matplotlib.pyplot as plt                # plotting library
from pathlib import Path

### import personal libraries
import linear_library as lib

# options to make the movie
movie = True                                   # switch to make an animation
movie_name = 'output_files/linear_wave_eqn_movie.mp4'

# save data
outfile = "output_files/linear_soln_data.npy"
outfile_spec = "output_files/linear_spec_data.npy"

### Input parameters
L    = 0.961                                   # length of domain                
N    = 25                                      # number of grid points
dx   = L/N                                     # grid spacing
c2_t = 1.13e5                                  # transverse wave speed (squared)

t0, tf  = 0, 0.1                             # initial time, final time
dt, ts  = 1e-10, 5e-6                          # time steps soln and output
m       = 1e5                                  # multiplication factor for tp and movie
tp      = dt*m                                 # time step for plotting

### Compute Parameters
Nt  = int(tf/dt)                               # mumber of time steps
npt = int(tp/dt)                               # mumber of time steps to plot
nsv = int(ts/dt)                               # mumber of time steps to save

kt = lib.compute_k(N, dx, np.sqrt(c2_t))

### Make output directories
Path("figures/").mkdir(parents=True, exist_ok=True)
Path("output_files/").mkdir(parents=True, exist_ok=True)

### Store parameters in a class then output some info
parms = lib.parameters(N = N, L = L, dx = dx, \
                   dt = dt, tf = tf, ts = ts, m = m, Nt = Nt, npt = npt, nsv = nsv, skip = 5, \
                   c2_t = c2_t, kt = kt, method = lib.flux_wave_eqn)
lib.output_info(parms)

### Initial Conditions with plot: uL, vL
x    = np.linspace(0, L, N+1)          # define grids (staggered grid)
xs   = np.linspace(0 + dx/2, L - dx/2, N)     # staggered grid
# ICs: initial velocity
# 0.5 -> 0
soln = np.vstack([0*x, 15.0*np.exp(-((x-L/2)**2)/(L/20)**2)])

### Store data to plot later
soln_save = np.zeros((2, N+1, round(tf/ts) + 1))
soln_save[:,:,0] = soln

spec_save = np.zeros((2, N, round((tf - t0) / tp) + 1))  # array to save spectrum values
spec = lib.fhat_all(soln, N)
spec_save[:, :, 0] = spec

### Start plotting snapshots    
fig = plt.figure(constrained_layout=True)
axs = fig.subplot_mosaic([["TopLeft", "Right"], ["BottomLeft", "Right"]])
lib.plot_soln(x, xs, soln, spec, parms, fig, axs, movie, 0)

### Calculate the solution
soln_save, spec_save = lib.calculate_soln(x, xs, soln, soln_save, spec_save, parms, fig, axs, movie)
plt.savefig("figures/linear_final_displacement.png")
plt.show()

### Save the data into a file
with open(outfile, "wb") as f:
    np.save(f, soln_save)

with open(outfile_spec, "wb") as f:
    np.save(f, spec_save)

### Make animation
if movie:
    lib.merge_to_mp4('figures/frame_%04d.png', movie_name)

## Hovmoller plots of the solution and save
lib.plot_hovmoller(x, soln_save, parms)