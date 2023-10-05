import numpy as np
from scipy.io import wavfile
import sys
import pywt

def splice_data_time(data, var, point):
    """ gets the values of the data for a specific variable
    and at one point for all the time values
    """
    return data[var, point, :]


if __name__ == "__main__":
    ### DATA FILE ###
    data = np.load("output_files/linear_soln_data.npy")

    ### PHYSICAL CONSTANTS ###
    # get time points -> maybe find a way to include it in parms?
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
    # we have data.shape[2] samples in duration seconds
    # so samplerate should be...
    samplerate = int(data.shape[2] / duration)  # number of samples per second?

    # let's start with uL
    uL_data = splice_data_time(data, 0, point) * 1e7

    # TODO: need to set up a plot -> y max should be abt 30k

    # ok and then... i think i should just be able to write it as a wav file?
    filename = f"audio_files/linear_piano.wav"
    wavfile.write(filename, samplerate, uL_data.astype(np.int16))

