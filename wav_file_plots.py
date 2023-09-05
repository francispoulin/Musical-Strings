#!/usr/bin/env python

""" plot a spectrogram of a wav file and isolate the initial time
the hope was to be able to use one time instant as initial conditions for nonlinear_wave_eqn.py
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

def plot_spectrogram(Fs, aud, note):
    """ plots a spectrogram of a wav file """
    Pxx, freq, t = mlab.specgram(aud, Fs=Fs)
    dbvals = 10 * np.log10(Pxx)

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(t, freq, dbvals, cmap='PiYG_r', shading="nearest")
    ax.axis('tight')
    ax.set(title=f"Spectrogram of {note}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ybound([0, 10000])
    #ax.set_yscale("log")
    fig.colorbar(mesh, label="Amplitude (dB)")

    return fig, Pxx, freq, t, dbvals

def plot_time_instant(freq, Pxx, i):
    fig = plt.figure()
    plt.plot(freq, Pxx.T[i, :], color="deeppink")
    plt.xlim([0, 10000])
    #plt.xscale("log")
    plt.title(f"Time: {t[i]} s")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")

    return fig

###

if __name__ == "__main__":

    file = "C:\\Users\\flutt\\Downloads\\Piano recordings\\Piano F.wav"
    note = "F4"

    # read file and generate the data
    Fs, aud = wavfile.read(file)

    # plot specgram manually
    fig, Pxx, freq, t, dB = plot_spectrogram(Fs, aud, note)

    # plot one time instant
    print(t[470:490])
    #for i in range(476, 490):
    #    fig = plot_time_instant(freq, dB, i)

    fig = plot_time_instant(freq, dB, 482)

    # display plots
    plt.show()