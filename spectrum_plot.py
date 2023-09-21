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
    plt.plot(time, splice_data_time(data, index[0], point), "b-", label="linear")
    plt.plot(time, splice_data_time(data, index[1], point), "r--", label="nonlinear")
    plt.plot(time, splice_data_time(data, index[2], point), "g-.", label="timoshenko")
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

if __name__ == "__main__":

    ### DATA FILE ###
    data = np.load("soln_data.npy")


    ### PHYSICAL CONSTANTS ###
    # get time points -> maybe find a way to include it in parms?
    t0 = 0.0
    tf = 2e-3
    time = np.linspace(t0, tf, data.shape[2])

    # choose our point of interest
    point = 25  # index 25 is the middle

    # for uL
    L    = 0.961                
    N    = 50
    dx   = L/N
    c_t = np.sqrt(1.13e5)
    c_l = np.sqrt(2.55e7)


    ### PLOTS ###

    # timeseries
    fig1 = timeseries(time, data, point, "Transverse")
    fig2 = timeseries(time, data, point, "Longitudinal")

    # spectrum
    spectrum_plot = plot_spectrum(data, "transverse", int(data.shape[2]/2), N, dx, c_t)

    #plt.show()


    ### SOUND ###
    # for the wav file, we need a sample rate
    # sample rate is in samples/sec units
    duration = tf - t0
    # we have data.shape[2] samples in duration seconds
    # so samplerate should be...
    samplerate = int(data.shape[2] / duration)  # number of samples per second?
    print(f"samplerate: {samplerate}")  # 2000500, in piano project it's 44100 so that's a HUGE difference...

    # let's say i want to artificially extend the sound so it's 1s long
    # then...
    samplerate = int(data.shape[2] / 1.0)


    # ok so for piano project what i did was
    # got all the freq & amplitude values (in dB)
    # then generated all the sin waves as amp*sin(2*pi*freq * t)
    # concatenated them all for each time interval and stitched them together basically
    # but i should be able to get what the wave looks like already at a certain point
    # from the timeseries plot
    # we choose the point on the string above

    # let's start with uL
    uL_data = splice_data_time(data, 0, point) * 1e8

    print(max(uL_data))

    # ok and then... i think i should just be able to write it as a wav file?
    filename = f"maybe_piano.wav"
    write(filename, samplerate, uL_data.astype(np.int16))
