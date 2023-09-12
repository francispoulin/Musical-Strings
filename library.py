#!/usr/bin/env python

import glob
import numpy as np                     # Import Libraries
import matplotlib.pyplot as plt
import os                              # to delete the png files
import subprocess                      # needed for movie
import sys

class parameters:
    def __init__(self, N, L, dx, dt, tf, ts, m, Nt, npt, nsv, skip, c2_t, c2_l, k, C1, C2, method):
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
        self.c2_l = c2_l
        self.k = k
        self.C1 = C1
        self.C2 = C2
        self.method = method

def merge_to_mp4(frame_filenames, movie_name, fps=12):

    f_log = open("ffmpeg.log", "w")
    f_err = open("ffmpeg.err", "w")
    cmd = ['ffmpeg', '-framerate', str(fps), '-i', frame_filenames, '-y', 
            '-q', '1', '-threads', '0', '-pix_fmt', 'yuv420p', movie_name]
    subprocess.call(cmd, stdout=f_log, stderr=f_err)
    # remove the frame PNGs
    for f in glob.glob("frame_*.png"):
        os.remove(f)
    f_log.close()
    f_err.close()

def dfdx(f,dx):                         # A difference function (positive direction)
    return (f[1:] - f[0:-1])/dx
  
def flux_wave_eqn(soln, parms):
    dx = parms.dx           
    c2_t = parms.c2_t
    c2_l = parms.c2_l
    k = parms.k
    C1 = parms.C1
    C2 = parms.C2
    N = parms.N

    #uL, vL, wL, sL, uN, vN, wN, sN, uT, vT, wT, sT, pT, vpT
    # 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13

    dudx_L = dfdx(soln[0, :], dx)
    dwdx_L = dfdx(soln[2, :], dx)
    dudx_N = dfdx(soln[4, :], dx)
    dwdx_N = dfdx(soln[6, :], dx)
    dRdx_N = np.sqrt(np.square((1+dwdx_N)) + np.square((dudx_N)))

    dudx_T = dfdx(soln[8, :], dx)
    dwdx_T = dfdx(soln[10, :], dx)
    dRdx_T = np.sqrt(np.square((1+dwdx_T)) + np.square((dudx_T)))
    dpdx_T = np.hstack([0, dfdx(soln[12, :-1], dx), 0])

    vec = soln[12, :-1] - dudx_T
    
    flux_vL = np.zeros(N+1)
    flux_sL = np.zeros(N+1)
    flux_vN = np.zeros(N+1)
    flux_sN = np.zeros(N+1)
    flux_vT = np.zeros(N+1)
    flux_sT = np.zeros(N+1)
    flux_vpT = np.zeros(N+1)

    flux_vL[1:-1] = c2_t * dfdx(dudx_L, dx) 
    flux_sL[1:-1] = c2_l * dfdx(dwdx_L, dx) 
    flux_vN[1:-1] = c2_l * dfdx(dudx_N, dx) - (c2_l - c2_t) * dfdx(np.divide(dudx_N, dRdx_N), dx)
    flux_sN[1:-1] = c2_l * dfdx(dwdx_N, dx) - (c2_l - c2_t) * dfdx(np.divide((1+dwdx_N), dRdx_N), dx)
    flux_vT[1:-1] = c2_l * dfdx(dudx_T, dx) - (c2_l - c2_t) * dfdx(np.divide(dudx_T, dRdx_T), dx) - C2 * dfdx(vec, dx)
    flux_sT[1:-1] = c2_l * dfdx(dwdx_T, dx) - (c2_l - c2_t) * dfdx(np.divide((1+dwdx_T), dRdx_T), dx)
    flux_vpT[:-1] = c2_l * dfdx(dpdx_T, dx) - C1 * C2 * vec


    flux = np.vstack([soln[1, :], flux_vL, soln[3, :], flux_sL, soln[5, :], flux_vN, soln[7, :], flux_sN, soln[9, :], flux_vT, soln[11, :], flux_sT, soln[13, :], flux_vpT])
              
    return flux

def plot_soln(x, xs, soln, parms, fig, axs, movie, ii):  # TODO
    L  = parms.L
    N  = parms.N
    m = parms.m
    sk = parms.skip

    axs[0, 0].cla()
    axs[0, 1].cla()
    axs[0, 2].cla()
    axs[1, 0].cla()
    axs[1, 1].cla()
    axs[1, 2].cla()

    t = ii*parms.dt
    fig.suptitle('Displacements in Elastic Rod at t = %7.7f' % t)

    axs[0, 0].plot(x, soln[0, :], '-b', linewidth=3, label="linear")  # linear u
    axs[0, 0].plot(x, soln[4, :], '--r', linewidth=3, label="nonlinear")  # nonlinear u
    axs[0, 0].plot(x, soln[8, :], '-.g', linewidth=3, label="Timoshenko")
    axs[0, 0].set_xlim([0, L])
    axs[0, 0].set_title("Transverse Displacements")
    #axs[0, 0].set_ylim([-1.7, 1.7])
    axs[0, 0].grid(True);
    axs[0, 0].legend(loc="best")

    axs[1, 0].plot(x, soln[1, :], '-b', linewidth=3, label="linear")  # linear v
    axs[1, 0].plot(x, soln[5, :], '--r', linewidth=3, label="nonlinear")  # nonlinear v
    axs[1, 0].plot(x, soln[9, :], '-.g', linewidth=3, label="Timoshenko")
    axs[1, 0].set_title("Transverse Velocities")
    #axs[1, 0].set_ylim([-1.8, 1.8])
    axs[1, 0].grid(True);
    axs[1, 0].legend(loc="best")

    axs[0, 1].plot(x, soln[2, :], '-b', linewidth=3, label="linear")  # linear u (long)
    axs[0, 1].plot(x, soln[6, :], '--r', linewidth=3, label="nonlinear")  # nonlinear u (long)
    axs[0, 1].plot(x, soln[10, :], '-.g', linewidth=3, label="Timoshenko")
    axs[0, 1].set_title("Longitudinal Displacements")
    #axs[0, 1].set_ylim([-9e-5, 9e-5])
    axs[0, 1].grid(True);
    axs[0, 1].legend(loc="best")

    axs[1, 1].plot(x, soln[3, :], '-b', linewidth=3, label="linear")  # linear v (long)
    axs[1, 1].plot(x, soln[7, :], '--r', linewidth=3, label="nonlinear")  # nonlinear v (long)
    axs[1, 1].plot(x, soln[11, :], '-.g', linewidth=3, label="Timoshenko") # timoshenko
    axs[1, 1].set_title("Longitudinal Velocities")
    #axs[1, 1].set_ylim([-0.8, 0.8])
    axs[1, 1].grid(True);
    axs[1, 1].legend(loc="best")

    axs[0, 2].plot(xs, soln[12, :-1], '-.g', linewidth=3, label="Timoshenko")  # pT
    axs[0, 2].set_title("Varphi Displacements")
    #axs[0, 2].set_ylim([-5e-2, 5e-2])
    axs[0, 2].grid(True);
    axs[0, 2].legend(loc="best")

    axs[1, 2].plot(xs, soln[13, :-1], '-.g', linewidth=3, label="Timoshenko")  # vpT
    axs[1, 2].set_title("Varphi Velocities")
    #axs[1, 2].set_ylim([-25, 25])
    axs[1, 2].grid(True);
    axs[1, 2].legend(loc="best")

    # hide inner tick labels
    #for ax in axs.flat:
    #    ax.label_outer()

    plt.draw()
    plt.pause(0.01)

    if movie:
        plt.savefig('frame_{0:04d}.png'.format(int(ii/parms.m)), dpi=200)


def plot_hovmoller(x, soln_save, parms):
    nsave = round(parms.tf/parms.ts) + 1
    fig, axs = plt.subplots(1, 2, sharey=True)     # Hovmoller plots of displacements
    fig.suptitle("Hovmoller plots of wave eqns")
    tts  = np.arange(nsave)*parms.ts
    hplot1 = axs[0].pcolor(x, tts, np.transpose(soln_save[0,:,:]))
    fig.colorbar(hplot1, ax=axs[0], extend='max')
    axs[0].set_xlim([-parms.L/2, parms.L/2])
    axs[0].set_ylim([tts[0], tts[-1]])
    axs[0].set_title('uL (transverse displacement linear)')

    hplot2 = axs[1].pcolor(x, tts, np.transpose(soln_save[4,:,:]))
    fig.colorbar(hplot2, ax=axs[1], extend='max')
    axs[1].set_xlim([-parms.L/2, parms.L/2])
    axs[1].set_ylim([tts[0], tts[-1]])
    axs[1].set_title('uN (transverse displacement nonlinear)')
    plt.savefig("hovmoller_plot_displacement.png")

def output_info(parms):
    cfl  = max(parms.c2_l, parms.c2_t)*parms.dt/parms.dx

    print("Solution to the one-dimensional wave equations")
    print("==============================================")
    print("Parameters (space):   L = ", parms.L, " N = ", parms.N, " dx = ", parms.dx)
    print("Parameters (time):    tf = ", parms.tf, "  dt = ", parms.dt, " ts = ", parms.ts)
    print(" ")
    print("Parameters (speed):      c2_t = ", parms.c2_t, " c2_l = ", parms.c2_l)
    print("Parameters (Tim. shear): k = ", parms.k)
    print("Parameters (constants):  C1 (A/I) = ", parms.C1, " C2 (Gk/rho) = ", parms.C2)
    print('Parameters (cfl):        cfl = ', cfl)
    print(" ")
    print("Plots will show ")
    print("     1) displacements: uL, uN, wL, wN, pN")
    print("     2) velocities:    vL, vN, sL, sN, vpN   ")

    if cfl > 0.5:
        print("The CFL paramter is greater than one.")
        print("Please try again but reduce dt so that CFL < -0.5")
        sys.exit('Stopping code!')

def calculate_soln(x, xs, soln, soln_save, parms, fig, axs, movie):

    dt = parms.dt
    method = parms.method

    NLnm = method(soln, parms)           # Euler step
    soln = soln + dt*NLnm;

    NLn  = method(soln, parms)           # AdamsBashforth2 step
    soln = soln + 0.5*dt*(3*NLn - NLnm)

    count_save = 1
    for ii in range(3,parms.Nt+3):                       # AdamsBashforth3 step
        t    = ii*dt   
        NL   = method(soln, parms);
        soln = soln + dt/12*(23*NL - 16*NLn + 5*NLnm)

        if ii%parms.npt==0:
            print('Plot at t = %6.7f' % t)
            plot_soln(x, xs, soln, parms, fig, axs, movie, ii)
    
        if ii%parms.nsv==0:
            soln_save[:,:,count_save] = soln[:,:]
            count_save += 1

        NLnm, NLn = NLn, NL                        # Reset fluxes
    return soln_save
