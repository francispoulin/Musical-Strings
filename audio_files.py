import numpy as np
from scipy.io import wavfile
import pywt

def splice_data_time(data, var, point):
    """ gets the values of the data for a specific variable
    and at one point for all the time values
    """
    return data[var, point, :]


if __name__ == "__main__":
    ### DATA FILE ###
    data = np.load("soln_data.npy")
    #spec_data = np.load("spec_data.npy")

    #print(data.shape)
    #print(spec_data.shape)


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

    ### SOUND ###
    # for the wav file, we need a sample rate
    # sample rate is in samples/sec units
    duration = tf - t0
    # we have data.shape[2] samples in duration seconds
    # so samplerate should be...
    samplerate = int(data.shape[2] / duration)  # number of samples per second?
    print(f"samplerate: {samplerate}")  # 2000500, in piano project it's 44100 so that's a HUGE difference...


    # ok so for piano project what i did was
    # got all the freq & amplitude values (in dB)
    # then generated all the sin waves as amp*sin(2*pi*freq * t)
    # concatenated them all for each time interval and stitched them together basically
    # but i should be able to get what the wave looks like already at a certain point
    # from the timeseries plot
    # we choose the point on the string above

    # let's start with uL
    uL_data = splice_data_time(data, 0, point)

    # then i think we need to do an inverse transform??? hmmm
    # well. let's just. try it out.
    # oh... i need cA and cD... but idk how to get those

    # ok and then... i think i should just be able to write it as a wav file?
    filename = f"maybe_piano.wav"
    wavfile.write(filename, samplerate, uL_data.astype(np.int16))
