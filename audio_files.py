#!/usr/bin/env python

""" script to generate an audio file from the output of nonlinear_wave_eqn.py or linear_wave_eqn.py """

import numpy as np
from scipy.io import wavfile

def splice_data_time(data, var, point):
    """ gets the values of the data for a specific variable
    and at one point for all the time values
    """
    return data[var, point, :]


if __name__ == "__main__":
    ### DATA FILE ###
    data = np.load("output_files/linear_soln_data.npy")

    ### PHYSICAL CONSTANTS ###
    t0 = 0.0
    tf = 0.1
    time = np.linspace(t0, tf, data.shape[2])

    # choose our point of interest
    point = 12

    # for uL
    L    = 0.961                
    N    = 25
    dx   = L/N
    c_t = np.sqrt(1.13e5)
    c_l = np.sqrt(2.55e7)    

    ### SOUND ###
    # for the wav file, we need a sample rate
    # sample rate is in samples/sec units
    duration = tf - t0
    samplerate = int(data.shape[2] / duration)

    # let's start with uL
    uL_data = splice_data_time(data, 0, point) * 1e7

    # NOTE: for future tests, it is good to PLOT uL_data or whatever you're saving
    # the max value should be ~30k
    # if it is too low then the sound will be inaudible, and if it is too high it will be static

    # save to wav file
    filename = f"audio_files/linear_piano.wav"
    wavfile.write(filename, samplerate, uL_data.astype(np.int16))

