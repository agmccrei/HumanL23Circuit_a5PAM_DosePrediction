################################################################################
# Import, Set up MPI Variables, Load Necessary Files
################################################################################
import os
from os.path import join
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.ticker import (LogLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.ticker as mticker
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy
from scipy import signal as ss
from scipy import stats as st
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import LFPy
import pandas as pd
from fooof import FOOOF

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':26})

N_cells = 1000
rate_threshold = 0.2 # Hz
dt = 0.025 # ms
startslice = 1000 # ms
endslice = 25000 # ms
t1 = int(startslice*(1/dt))
t2 = int(endslice*(1/dt))
tvec = np.arange(endslice/dt+1)*dt
nperseg = 80000 # len(tvec[t1:t2])/2
fmaxval = 250
sampling_rate = (1/dt)*1000

radii = [79000., 80000., 85000., 90000.]
sigmas = [0.47, 1.71, 0.02, 0.41] #conductivity
L23_pos = np.array([0., 0., 78200.]) #single dipole refernece for EEG/ECoG
EEG_sensor = np.array([[0., 0., 90000]])

EEG_args = LFPy.FourSphereVolumeConductor(radii, sigmas, EEG_sensor)

state_seeds = [i for i in range(1,11)]
circuit_seed = 10234

paths = ['Seeds_FullSet_withEEG/Circuit_output/DIPOLEMOMENT_CircuitSeed'+str(circuit_seed)+'StimSeed'+str(i)+'.npy' for i in state_seeds]

for cind1, path1 in enumerate(paths):
	eeg = []
	print('Analyzing circuit seed #'+str(circuit_seed)+' for state seed #'+str(state_seeds[cind1]))
	# Load outputs
	try:
		temp_e = np.load(path1)
	except:
		print('Folder not found for circuit seed #'+str(circuit_seed)+' / state seed #'+str(state_seeds[cind1]))
		continue
	
	# EEG
	temp_e2 = temp_e['HL23PYR']
	temp_e2 = np.add(temp_e2,temp_e['HL23SST'])
	temp_e2 = np.add(temp_e2,temp_e['HL23PV'])
	temp_e2 = np.add(temp_e2,temp_e['HL23VIP'])
	
	potential = EEG_args.calc_potential(temp_e2, L23_pos)
	EEG = potential[0][t1:t2]
	
	freqrawEEG, psrawEEG = ss.welch(EEG, sampling_rate, nperseg=nperseg)
	fmaxind = np.where(freqrawEEG>=fmaxval)[0][0]
	
	fig = plt.figure(figsize=(8,7))
	ax = fig.add_subplot(111)
	ax.plot(freqrawEEG, psrawEEG, c='k')
	ax.set_xlim(3,30)
	ax.set_ylabel('Power')
	ax.set_xlabel('Frequency (Hz)')
	fig.savefig('Saved_PSDs_AllSeeds/Plots/EEG_PSD_CircuitSeed' + str(circuit_seed) + 'StimSeed'+str(state_seeds[cind1]),bbox_inches='tight', dpi=300, transparent=True)
	
	np.save('Saved_PSDs_AllSeeds/CircuitSeed' + str(circuit_seed) + 'StimSeed' + str(state_seeds[cind1]) + '_EEG',psrawEEG[:fmaxind])
