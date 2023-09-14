#!/usr/bin/env python

""" script to generate plots from the output file of nonlinear_wave_eqn.py
file should be a .npy file with the following indices:
    first index: variable
    second index: point on the string
    third index: time

    variables:
        uL, vL, wL, sL, uN, vN, wN, sN, uT, vT, wT, sT, pT, vpT
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13
    u = transverse displacement; v = transverse velocity
    w = longitudinal displacement; s = longitudinal velocity
    p = varphi displacement; vp = varphi velocity (Timoshenko model)
    L = linear; N = nonlinear; T = Timoshenko model
"""

# it would be nice if there was a config file or smth so that we could get the parms class in this file too
# maybe smth simple woould be easy to make?

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import sys

def timeseries(time, data, point, type):
    """ type must be either 'Transverse' or 'Longitudinal'"""
    if type == "Transverse":
        index = [0, 4, 8]
    elif type == "Longitudinal":
        index = [2, 6, 10]
    else:
        print("Type is not 'Transverse' or 'Longitudinal'. Exiting.")
        sys.exit()
    fig = plt.figure()
    plt.plot(time, data[index[0], point, :], "b-", label="linear")
    plt.plot(time, data[index[1], point, :], "r--", label="nonlinear")
    plt.plot(time, data[index[2], point, :], "g-.", label="timoshenko")
    plt.legend()
    plt.title(f"{type} displacement over time")
    return fig

def compute_k(N, dx, c):
    kodd = fftpack.fftfreq(2*N, d=dx)
    freq = kodd*c
    freqh = freq[0:N]

    return freqh

def compute_fhat(data, var, time, N):
    """ var: int (1st index of array)
        time: int (last index of array)
    """
    f = data[var, :, time]
    fodd = np.hstack([f, 0, -np.flipud(f[1:])])
    fodd_hat = fftpack.fft(fodd)
    fhat = fodd_hat[0:N]

    return fhat

def plot_spectrum(data, type, time, N, dx, c):
    k = compute_k(N, dx, c)
    
    if type == "transverse":
        index = [0, 4, 8]
    elif type == "longitudinal":
        index = [2, 6, 10]
    else:
        print("Sorry, type given is not recognized!")
        sys.exit()

    fL = compute_fhat(data, index[0], time, N)
    fN = compute_fhat(data, index[1], time, N)
    fT = compute_fhat(data, index[2], time, N)

    fig = plt.figure()
    plt.plot(k, np.abs(fL), color="deeppink", marker=".", linestyle="-", label="linear")
    plt.plot(k, np.abs(fN), color="dodgerblue", marker=".", linestyle="--", label="nonlinear")
    plt.plot(k, np.abs(fT), color="goldenrod", marker=".", linestyle="-.", label="timoshenko")
    plt.xlabel("k*c values (Hz)")
    plt.ylabel("f hat values")
    plt.title(f"Spectrum plot of transverse waves at time {time}")
    plt.legend()
    
    return fig

###

data = np.load("soln_data.npy")

# get time points -> maybe find a way to include it in parms?
t0 = 0.0
tf = 1e-4
time = np.linspace(t0, tf, data.shape[2])

# timeseries
fig1 = timeseries(time, data, 25, "Transverse")
fig2 = timeseries(time, data, 25, "Longitudinal")

#plt.show()

# for uL
L    = 0.961                
N    = 50
dx   = L/N
c_t = np.sqrt(1.13e5)
c_l = np.sqrt(2.55e7)

# think abt integrating this into nonlinear_wave_eqn.py
# to plot all of them at the same time (2x4 or 2x5)

# maybe use a different dx?
# https://stackoverflow.com/questions/9456037/scipy-numpy-fft-frequency-analysis

#x    = np.linspace(0, L, N)
#xodd = np.linspace(0, 2*L-dx, 2*N)

#f_L = data[0, :, int(data.shape[2]-1)]
#fodd = np.hstack([f_L, 0, -np.flipud(f_L[1:])])
#fodd_hat = fftpack.fft(fodd)
#f_hath_L = fodd_hat[0:N]

#f_hath_L = compute_fhat(data, 0, int(data.shape[2]-1), N)
#f_hath_N = compute_fhat(data, 4, int(data.shape[2]-1), N)
#f_hath_T = compute_fhat(data, 8, int(data.shape[2]-1), N)

#f_N = data[4, :, int(data.shape[2]-1)]
#fodd = np.hstack([f_N, 0, -np.flipud(f_N[1:])])
#fodd_hat = fftpack.fft(fodd)
#f_hath_N = fodd_hat[0:N]

#f_T = data[8, :, int(data.shape[2]-1)]
#fodd = np.hstack([f_T, 0, -np.flipud(f_T[1:])])
#fodd_hat = fftpack.fft(fodd)
#f_hath_T = fodd_hat[0:N]

#kodd = fftpack.fftfreq(2*N, d=dx)
#freq = kodd*c_t
#freqh = freq[0:N]

#freqh = compute_k(N, dx, c_t)

#fig = plt.figure()
#plt.plot(freqh, np.abs(f_hath_L), color="deeppink", marker=".", linestyle="-", label="linear")
#plt.plot(freqh, np.abs(f_hath_N), color="dodgerblue", marker=".", linestyle="--", label="nonlinear")
#plt.plot(freqh, np.abs(f_hath_T), color="goldenrod", marker=".", linestyle="-.", label="timoshenko")
#plt.xlabel("k*c values (Hz)")
#plt.ylabel("f hat values")
#plt.title(f"Spectrum plot of transverse waves at time {time[int(data.shape[2]/2)]}")
#plt.legend()
#plt.show()

spectrum_plot = plot_spectrum(data, "transverse", int(data.shape[2]/2), N, dx, c_t)

plt.show()


xs   = np.linspace(0 + dx/2, L - dx/2, N)
