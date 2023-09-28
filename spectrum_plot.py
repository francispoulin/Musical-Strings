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
from scipy.io.wavfile import write
import sys

def splice_data_time(data, var, point):
    """ gets the values of the data for a specific variable
    and at one point for all the time values
    """
    return data[var, point, :]

def timeseries(time, data, point, ax=None):

    index = {}
    index[0] = [0, 4, 8]
    index[1] = [2, 6, 10]

    if ax is None:
        ret = True
        fig, ax = plt.subplots(2, 1, sharex=True)
        plt.suptitle(f"Displacement vs. Time at Point {point}")
    else:
        ret = False
        plt.suptitle("Displacement vs. Time")

    ax[0].set_title("Transverse Displacement")
    ax[1].set_title("Longitudinal Displacement")
    
    for i in [0, 1]:
        ax[i].plot(time, splice_data_time(data, index[i][0], point), "b-", label="linear")
        ax[i].plot(time, splice_data_time(data, index[i][1], point), "r--", label="nonlinear")
        ax[i].plot(time, splice_data_time(data, index[i][2], point), "g-.", label="timoshenko")
        ax[i].set_ylabel("Displacement (m)")

    plt.xlabel("Time (s)")

    if ret:
        ax[0].legend()
        ax[1].legend()
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

if __name__ == "__main__":

    ### DATA FILE ###
    data = np.load("output_files/soln_data_002.npy")
    #spec_data = np.load("spec_data.npy")

    #print(data.shape)
    #print(spec_data.shape)


    ### PHYSICAL CONSTANTS ###
    # get time points -> maybe find a way to include it in parms?
    t0 = 0.0
    tf = 2e-3
    time = np.linspace(t0, tf, data.shape[2])

    # for uL
    L    = 0.961                
    N    = 50
    dx   = L/N
    c_t = np.sqrt(1.13e5)
    c_l = np.sqrt(2.55e7)


    ### PLOTS ###

    # timeseries
    fig = timeseries(time, data, 25)
    plt.savefig("figures/timeseries.png")

    # timeseries but all points of the string on one plot
    fig, ax = plt.subplots(2, 1, sharex=True)
    for point in np.arange(N):
        fig = timeseries(time, data, point, ax)

    plt.savefig("figures/timeseries_all.png")

    plt.show()

    # spectrum
    time_index = int(data.shape[2]/2)
    spectrum_plot = plot_spectrum(data, "transverse", time_index, N, dx, c_t)



