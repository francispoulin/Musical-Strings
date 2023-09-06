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

import numpy as np
import matplotlib.pyplot as plt
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

###

data = np.load("soln_data.npy")

# get time points -> maybe find a way to include it in parms?
t0 = 0.0
tf = 1e-4
time = np.linspace(t0, tf, data.shape[2])

# timeseries
#fig1 = timeseries(time, data, 25, "Transverse")
#fig2 = timeseries(time, data, 25, "Longitudinal")

#plt.show()

