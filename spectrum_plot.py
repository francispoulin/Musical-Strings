import numpy as np
import matplotlib.pyplot as plt

data = np.load("soln_data.npy")

print(data.shape)

#uL, vL, wL, sL, uN, vN, wN, sN, uT, vT, wT, sT, pT, vpT
# 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13

# first index is the variable
# second index is the point on the string
# third index is the time

# let's start by making a time series for one point (maybe the middle point?)

# get time points
t0 = 0.0
tf = 0.001
time = np.linspace(t0, tf, data.shape[2])

fig = plt.figure()
plt.plot(time, data[0, 25, :], "b-", label="linear")
plt.plot(time, data[4, 25, :], "r--", label="nonlinear")
plt.plot(time, data[8, 25, :], "g-.", label="timoshenko")
plt.legend()
plt.title("Transverse displacement over time")

fig = plt.figure()
plt.plot(time, data[2, 25, :], "b-", label="linear")
plt.plot(time, data[6, 25, :], "r--", label="nonlinear")
plt.plot(time, data[10, 25, :], "g-.", label="timoshenko")
plt.legend()
plt.title("Longitudinal displacement over time")

plt.show()