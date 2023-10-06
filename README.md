MUSICAL-STRINGS

GOAL: reproduce the sound of a piano string by solving the wave equation numerically (was not fully successful)

SCRIPTS:
audio_files.py: script to generate wav files from x_wave_eqn.py
library.py: function library for nonlinear_wave_eqn.py
linear_library.py: function library for linear_wave_eqn.py
linear_wave_eqn.py: solves the linear wave equation on a string in one direction
nonlinear_wave_eqn.py: solves the linear wave eqn in two directions, the nonlinear wave eqn, and a Timoshenko model in two directions
spectrum_plot.py: script to generate output figures for nonlinear_wave_eqn.py and linear_wave_eqn.py
wav_file_plots.py: script to generate plots from an audio file (wav format)

REMAINING TODO ITEMS:
- spectrum_plot.py can be cleaned up, right now there is code for the linear wave eqn output separated from the code for the nonlinear wave eqn output by a sys.exit()
- dispersion may help explain how the piano notes start and end (not all at once and fade out)
- pianos actually have 3 strings per note (right now it computes for a single string)
- efficiency: nonlinear_wave_eqn takes 6 hours on my (Sarah's) PC to compute 0.002s of sound
- inharmonicity: nonlinear wave eqn with Timoshenko model may account for inharmonicity of piano harmonics