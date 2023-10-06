#!/usr/bin/env python

""" library functions for linear_wave_eqn.py 
NOTE that they are not the most efficient or cleanest because it was made in a bit of a rush
some variables are unneeded since there is only one wave equation
"""

import glob
from scipy import fftpack
import numpy as np                     # Import Libraries
import matplotlib.pyplot as plt
import os                              # to delete the png files
import subprocess                      # needed for movie
import sys

class parameters:
    def __init__(self, N, L, dx, dt, tf, ts, m, Nt, npt, nsv, skip, c2_t, kt, method):
        self.N  = N
        self.L  = L
        self.dx = dx
        self.dt = dt
        self.tf = tf
        self.ts = ts
        self.m = m
        self.Nt = Nt
        self.npt = npt
        self.nsv = nsv
        self.skip = skip
        self.c2_t = c2_t
        self.kt = kt
        self.method = method

def merge_to_mp4(frame_filenames, movie_name, fps=12):

    f_log = open("output_files/ffmpeg.log", "w")
    f_err = open("output_files/ffmpeg.err", "w")
    cmd = ['ffmpeg', '-framerate', str(fps), '-i', frame_filenames, '-y', 
            '-q', '1', '-threads', '0', '-pix_fmt', 'yuv420p', movie_name]
    subprocess.call(cmd, stdout=f_log, stderr=f_err)
    # remove the frame PNGs
    for f in glob.glob("figures/frame_*.png"):
        os.remove(f)
    f_log.close()
    f_err.close()

def dfdx(f,dx):                         # A difference function (positive direction)
    return (f[1:] - f[0:-1])/dx
  
def flux_wave_eqn(soln, parms):
    dx = parms.dx           
    c2_t = parms.c2_t
    N = parms.N

    dudx = dfdx(soln[0, :], dx)
    
    flux_v = np.zeros(N+1)

    flux_v[1:-1] = c2_t * dfdx(dudx, dx) 


    flux = np.vstack([soln[1, :], flux_v])
              
    return flux

def plot_soln(x, xs, soln, spec, parms, fig, axs, movie, ii):  # TODO
    L  = parms.L
    kt = parms.kt

    axs["TopLeft"].cla()
    axs["BottomLeft"].cla()
    axs["Right"].cla()

    t = ii*parms.dt
    fig.suptitle('Displacements in Elastic Rod at t = %7.7f' % t)

    axs["TopLeft"].plot(x, soln[0, :], 'dodgerblue', linewidth=3)
    axs["TopLeft"].set_xlim([0, L])
    axs["TopLeft"].set_title("Displacement")
    axs["TopLeft"].set_ylim([-0.0025, 0.0025])
    axs["TopLeft"].grid(True)

    axs["BottomLeft"].plot(x, soln[1, :], 'dodgerblue', linewidth=3)  # linear v
    axs["BottomLeft"].set_title("Velocity")
    axs["BottomLeft"].set_ylim([-16.0, 16.0])
    axs["BottomLeft"].grid(True)

    # power spectrum plots
    axs["Right"].plot(kt, spec[0, :], color="deeppink", marker=".", linestyle="-")
    axs["Right"].set_xlabel("frequency (Hz)")
    axs["Right"].set_ylabel("f hat")
    axs["Right"].set_title("Spectrum")
    axs["Right"].grid(True)

    plt.draw()
    plt.pause(0.01)

    if movie:
        plt.savefig('figures/frame_{0:04d}.png'.format(int(ii/parms.m)), dpi=200)


def plot_hovmoller(x, soln_save, parms):
    nsave = round(parms.tf/parms.ts) + 1
    fig, axs = plt.subplots(1, 2, sharey=True)     # Hovmoller plots of displacements
    fig.suptitle("Hovmoller plots of wave eqns")
    tts  = np.arange(nsave)*parms.ts

    hplot1 = axs[0].pcolor(x, tts, np.transpose(soln_save[0,:,:]))
    fig.colorbar(hplot1, ax=axs[0], extend='max')
    axs[0].set_xlim([-parms.L/2, parms.L/2])
    axs[0].set_ylim([tts[0], tts[-1]])
    axs[0].set_title('displacement')

    hplot2 = axs[1].pcolor(x, tts, np.transpose(soln_save[1,:,:]))
    fig.colorbar(hplot2, ax=axs[1], extend='max')
    axs[1].set_xlim([-parms.L/2, parms.L/2])
    axs[1].set_ylim([tts[0], tts[-1]])
    axs[1].set_title('velocity')

    plt.savefig("figures/linear_hovmoller_plot_displacement.png")

def output_info(parms):
    cfl  = parms.c2_t*parms.dt/parms.dx

    print("Solution to the one-dimensional wave equations")
    print("==============================================")
    print("Parameters (space):   L = ", parms.L, " N = ", parms.N, " dx = ", parms.dx)
    print("Parameters (time):    tf = ", parms.tf, "  dt = ", parms.dt, " ts = ", parms.ts)
    print(" ")
    print("Parameters (speed):      c2_t = ", parms.c2_t)
    print('Parameters (cfl):        cfl = ', cfl)
    print(" ")
    print("Plots will show ")
    print("     1) displacements: uL, uN, wL, wN, pN")
    print("     2) velocities:    vL, vN, sL, sN, vpN   ")

    if cfl > 0.5:
        print("The CFL paramter is greater than one.")
        print("Please try again but reduce dt so that CFL < -0.5")
        sys.exit('Stopping code!')

def compute_k(N, dx, c):
    kodd = fftpack.fftfreq(2*N, d=dx)
    freq = kodd*c
    freqh = freq[0:N]

    return freqh

def compute_fhat(soln, N):
    fodd = np.hstack([soln, 0, -np.flipud(soln[1:])])
    fodd_hat = fftpack.fft(fodd)
    fhat = fodd_hat[0:N]

    return fhat

def fhat_all(soln_array, N):
    """ i say all but really only the displacements """
    fhat = np.zeros((1, N))

    # transverse
    fhat[0, :] = np.abs(compute_fhat(soln_array[0, :], N))

    return fhat
    
def calculate_soln(x, xs, soln, soln_save, spec_save, parms, fig, axs, movie):

    dt = parms.dt
    method = parms.method

    NLnm = method(soln, parms)           # Euler step
    soln = soln + dt*NLnm;

    NLn  = method(soln, parms)           # AdamsBashforth2 step
    soln = soln + 0.5*dt*(3*NLn - NLnm)

    count_save = 1
    count_spec = 1
    for ii in range(3,parms.Nt+3):                       # AdamsBashforth3 step
        t    = ii*dt   
        NL   = method(soln, parms);
        soln = soln + dt/12*(23*NL - 16*NLn + 5*NLnm)

        if ii%parms.npt==0:
            # compute spectrum
            spec = fhat_all(soln, parms.N)
            spec_save[:, :, count_spec] = spec[:, :]
            count_spec += 1

            print('Plot at t = %6.7f' % t)
            plot_soln(x, xs, soln, spec, parms, fig, axs, movie, ii)
    
        if ii%parms.nsv==0:
            soln_save[:,:,count_save] = soln[:,:]
            count_save += 1

        NLnm, NLn = NLn, NL                        # Reset fluxes
    return soln_save, spec_save
