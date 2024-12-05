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
from scipy.optimize import curve_fit
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import LFPy
import pandas as pd
from fooof import FOOOF
from copy import copy
import itertools

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':26})

N_seeds = 20
N_seedsList = np.linspace(1,N_seeds,N_seeds, dtype=int)
N_cells = 1000
rate_threshold = 0.2 # Hz
dt = 0.025 # ms
startslice = 1000 # ms
endslice = 25000 # ms
t1 = int(startslice*(1/dt))
t2 = int(endslice*(1/dt))
tvec = np.arange(endslice/dt+1)*dt
nperseg = 80000 # len(tvec[t1:t2])/2

radii = [79000., 80000., 85000., 90000.]
sigmas = [0.47, 1.71, 0.02, 0.41] #conductivity
L23_pos = np.array([0., 0., 78200.]) #single dipole refernece for EEG/ECoG
EEG_sensor = np.array([[0., 0., 90000]])

EEG_args = LFPy.FourSphereVolumeConductor(radii, sigmas, EEG_sensor)

severities = np.array([0.0,0.1,0.2,0.3,0.4])
doses = np.array([0.00,0.25,0.50,0.75,1.00,1.25,1.50])

doses_of_interest = np.array([0.50,0.75,1.25]) # Should be of length severities - 1

conds = [['MDD_severity_'+f"{s:0.1f}"+'_a5PAM_'+f"{d:0.2f}"+'_long' for s in severities] for d in doses]
paths = [['Saved_PSDs_AllSeeds_nperseg_80000/'+i for i in c] for c in conds]

x_labels = [f"{d*100:0.0f}" + '%' for d in doses]
x_labels2 = [['Severity '+f"{s*100:0.1f}"+'% + Dose '+f"{d*100:0.2f}" + '%' for s in severities] for d in doses]
x_labels_types = ['Pyr', 'SST', 'PV', 'VIP']

# Create array of colors
def arr_creat(upperleft, upperright, lowerleft, lowerright):
	arr = np.linspace(np.linspace(lowerleft, lowerright, arrwidth),np.linspace(upperleft, upperright, arrwidth), arrheight, dtype=int)
	return arr[:, :, None]

arrwidth = len(severities)
arrheight = len(doses)

# just a reminder
dodgerblue = (30,144,255) # drug
purple = (148,103,189) # MDD
gray = (127,127,127) # healthy

r = arr_creat(30 , 30 , 127, 148)
g = arr_creat(144, 144, 127, 103)
b = arr_creat(255, 255, 127, 189)

colors_conds0 = np.concatenate([r, g, b], axis=2)
colors_conds = colors_conds0/255

plt.imshow(np.flipud(colors_conds0), origin="lower")
plt.axis("off")

plt.savefig('test_colors.png',dpi=300,transparent=True)
plt.close()

# Color condition array for when MDD only
lavenderblush = (255,240,245)
rebeccapurple = (102,51,153)

r = arr_creat(30 , 30 , 255, 75)
g = arr_creat(144, 144, 240, 0)
b = arr_creat(255, 255, 245, 130)

colors_conds0_MDDonly = np.concatenate([r, g, b], axis=2)
colors_conds_MDDonly = colors_conds0_MDDonly/255

colors_neurs = ['dimgrey', 'red', 'green', 'orange']

thetaband = (4,8)
alphaband = (8,12)
betaband = (12,21)
broadband_ab = (8,21)
broadband = (3,30)

def cohen_d(y,x):
	nx = len(x)
	ny = len(y)
	dof = nx + ny - 2
	return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

eeg = []
spikes = []
spikes_PN = []
for pind,p in enumerate(paths):
	eeg.append([])
	spikes.append([])
	spikes_PN.append([])
	for path in p:
		try:
			eeg[pind].append(np.load(path + '_EEG.npy'))
			spikes[pind].append(np.load(path + '_spikes.npy'))
			spikes_PN[pind].append(np.load(path + '_spikes_PN.npy'))
		except:
			eeg[pind].append(np.nan)
			spikes[pind].append(np.nan)
			spikes_PN[pind].append(np.nan)

offsets = [[[] for s in severities] for d in doses]
exponents = [[[] for s in severities] for d in doses]
knees = [[[] for s in severities] for d in doses]
errors = [[[] for s in severities] for d in doses]
AUC = [[[] for s in severities] for d in doses]

Ls = [[[] for s in severities] for d in doses]
Gns = [[[] for s in severities] for d in doses]

center_freqs_max = [[[] for s in severities] for d in doses]
power_amps_max = [[[] for s in severities] for d in doses]
bandwidths_max = [[[] for s in severities] for d in doses]
AUC_max = [[[] for s in severities] for d in doses]

center_freqs_t = [[[] for s in severities] for d in doses]
power_amps_t = [[[] for s in severities] for d in doses]
bandwidths_t = [[[] for s in severities] for d in doses]
AUC_t = [[[] for s in severities] for d in doses]
AUC_t_abs = [[[] for s in severities] for d in doses]

center_freqs_a = [[[] for s in severities] for d in doses]
power_amps_a = [[[] for s in severities] for d in doses]
bandwidths_a = [[[] for s in severities] for d in doses]
AUC_a = [[[] for s in severities] for d in doses]
AUC_a_abs = [[[] for s in severities] for d in doses]

center_freqs_b = [[[] for s in severities] for d in doses]
power_amps_b = [[[] for s in severities] for d in doses]
bandwidths_b = [[[] for s in severities] for d in doses]
AUC_b = [[[] for s in severities] for d in doses]
AUC_b_abs = [[[] for s in severities] for d in doses]

AUC_broad_abs = [[[] for s in severities] for d in doses]

base_rates = [[[] for s in severities] for d in doses]

def scalebar(axis,xy,lw=3):
	# xy = [left,right,bottom,top]
	xscalebar = np.array([xy[0],xy[1],xy[1]])
	yscalebar = np.array([xy[2],xy[2],xy[3]])
	axis.plot(xscalebar,yscalebar,'k',linewidth=lw)

for seed in N_seedsList:
	for cind1, path1 in enumerate(paths):
		for cind2, path2 in enumerate(path1):
			print('Analyzing seed #'+str(seed)+' for '+conds[cind1][cind2])
			
			if eeg[cind1][cind2] is np.nan:
				print(paths[cind1][cind2] + ' does not exist')
				offsets[cind1][cind2].append(np.nan)
				exponents[cind1][cind2].append(np.nan)
				errors[cind1][cind2].append(np.nan)
				Ls[cind1][cind2].append(np.nan)
				Gns[cind1][cind2].append(np.nan)
				AUC[cind1][cind2].append(np.nan)
				AUC_t[cind1][cind2].append(np.nan)
				AUC_a[cind1][cind2].append(np.nan)
				AUC_b[cind1][cind2].append(np.nan)
				AUC_broad_abs[cind1][cind2].append(np.nan)
				AUC_t_abs[cind1][cind2].append(np.nan)
				AUC_a_abs[cind1][cind2].append(np.nan)
				AUC_b_abs[cind1][cind2].append(np.nan)
				center_freqs_t[cind1][cind2].append(np.nan)
				power_amps_t[cind1][cind2].append(np.nan)
				bandwidths_t[cind1][cind2].append(np.nan)
				center_freqs_a[cind1][cind2].append(np.nan)
				power_amps_a[cind1][cind2].append(np.nan)
				bandwidths_a[cind1][cind2].append(np.nan)
				center_freqs_b[cind1][cind2].append(np.nan)
				power_amps_b[cind1][cind2].append(np.nan)
				bandwidths_b[cind1][cind2].append(np.nan)
				center_freqs_max[cind1][cind2].append(np.nan)
				power_amps_max[cind1][cind2].append(np.nan)
				bandwidths_max[cind1][cind2].append(np.nan)
				AUC_max[cind1][cind2].append(np.nan)
				base_rates[cind1][cind2].append(np.nan)
				continue
			
			try:
				temp_s = np.load('Saved_SpikesOnly/'+conds[cind1][cind2] + '/SPIKES_Seed' + str(seed) + '.npy',allow_pickle=True)
			except:
				print(conds[cind1][cind2] + ' does not exist')
				continue
			
			SPIKES = temp_s.item()
			temp_rn = []
			for i in range(0,len(SPIKES['times'][0])):
				scount = SPIKES['times'][0][i][(SPIKES['times'][0][i]>startslice) & (SPIKES['times'][0][i]<=endslice)]
				Hz = (scount.size)/((int(endslice)-startslice)/1000)
				temp_rn.append(Hz)
			
			base_rates[cind1][cind2].append(np.mean(temp_rn))
			
			f_res = 1/(nperseg*dt/1000)
			freqrawEEG = np.arange(0,101,f_res)
			
			frange_init = 3
			frange = [frange_init,30]
			fm = FOOOF(peak_width_limits=(2, 6.),
						min_peak_height=0,
						aperiodic_mode='fixed',
						max_n_peaks=3,
						peak_threshold=2.)
			
			fm.fit(freqrawEEG, eeg[cind1][cind2][seed-1], frange)
			
			offsets[cind1][cind2].append(fm.aperiodic_params_[0])
			exponents[cind1][cind2].append(fm.aperiodic_params_[-1])
			errors[cind1][cind2].append(fm.error_)
			
			L = 10**fm._ap_fit[np.where(fm.freqs>=frange_init)[0].tolist()[0]:]
			Gn = fm._peak_fit[np.where(fm.freqs>=frange_init)[0].tolist()[0]:]
			F = fm.freqs[np.where(fm.freqs>=frange_init)[0].tolist()[0]:]
			
			Ls[cind1][cind2].append(L)
			Gns[cind1][cind2].append(Gn)
			
			inds_ = [iids[0] for iids in np.argwhere((F>=broadband[0]) & (F<=broadband[1]))]
			inds_t = [iids[0] for iids in np.argwhere((F>=thetaband[0]) & (F<=thetaband[1]))]
			inds_a = [iids[0] for iids in np.argwhere((F>=alphaband[0]) & (F<=alphaband[1]))]
			inds_b = [iids[0] for iids in np.argwhere((F>=betaband[0]) & (F<=betaband[1]))]
			
			AUC[cind1][cind2].append(np.trapz(L[inds_],x=F[inds_]))
			AUC_t[cind1][cind2].append(np.trapz(Gn[inds_t],x=F[inds_t]))
			AUC_a[cind1][cind2].append(np.trapz(Gn[inds_a],x=F[inds_a]))
			AUC_b[cind1][cind2].append(np.trapz(Gn[inds_b],x=F[inds_b]))
			
			inds_ = [iids[0] for iids in np.argwhere((freqrawEEG>=broadband_ab[0]) & (freqrawEEG<=broadband_ab[1]))]
			inds_t = [iids[0] for iids in np.argwhere((freqrawEEG>=thetaband[0]) & (freqrawEEG<=thetaband[1]))]
			inds_a = [iids[0] for iids in np.argwhere((freqrawEEG>=alphaband[0]) & (freqrawEEG<=alphaband[1]))]
			inds_b = [iids[0] for iids in np.argwhere((freqrawEEG>=betaband[0]) & (freqrawEEG<=betaband[1]))]
			
			AUC_broad_abs[cind1][cind2].append(np.trapz(eeg[cind1][cind2][seed-1][inds_],x=freqrawEEG[inds_]))
			AUC_t_abs[cind1][cind2].append(np.trapz(eeg[cind1][cind2][seed-1][inds_t],x=freqrawEEG[inds_t]))
			AUC_a_abs[cind1][cind2].append(np.trapz(eeg[cind1][cind2][seed-1][inds_a],x=freqrawEEG[inds_a]))
			AUC_b_abs[cind1][cind2].append(np.trapz(eeg[cind1][cind2][seed-1][inds_b],x=freqrawEEG[inds_b]))
			
			prev_max = 0
			for ii in fm.peak_params_:
				cf = ii[0]
				pw = ii[1]
				bw = ii[2]
				if thetaband[0] <= cf <= thetaband[1]:
					center_freqs_t[cind1][cind2].append(cf)
					power_amps_t[cind1][cind2].append(pw)
					bandwidths_t[cind1][cind2].append(bw)
				if alphaband[0] <= cf <= alphaband[1]:
					center_freqs_a[cind1][cind2].append(cf)
					power_amps_a[cind1][cind2].append(pw)
					bandwidths_a[cind1][cind2].append(bw)
				if betaband[0] <= cf <= betaband[1]:
					center_freqs_b[cind1][cind2].append(cf)
					power_amps_b[cind1][cind2].append(pw)
					bandwidths_b[cind1][cind2].append(bw)
				if pw > prev_max: # find max peak amplitude
					cf_max = cf # Max peak center frequency
					pw_max = pw # Max peak amplitude
					bw_max = bw # Max peak bandwidth
					inds_max = [iids[0] for iids in np.argwhere((F>=cf-bw/2) & (F<=cf+bw/2))]
					aw_max = np.trapz(Gn[inds_max],x=F[inds_max]) # Max peak AUC (using center frequency and bandwidth)
					prev_max = pw
			center_freqs_max[cind1][cind2].append(cf_max)
			power_amps_max[cind1][cind2].append(pw_max)
			bandwidths_max[cind1][cind2].append(bw_max)
			AUC_max[cind1][cind2].append(aw_max)

# Area under frequency bands - bar plots
offsets_m = [[np.mean(allvals) for allvals in a] for a in offsets]
exponents_m = [[np.mean(allvals) for allvals in a] for a in exponents]
errors_m = [[np.mean(allvals) for allvals in a] for a in errors]
center_freqs_t_m = [[np.mean(allvals) for allvals in a] for a in center_freqs_t]
power_amps_t_m = [[np.mean(allvals) for allvals in a] for a in power_amps_t]
bandwidths_t_m = [[np.mean(allvals) for allvals in a] for a in bandwidths_t]
center_freqs_a_m = [[np.mean(allvals) for allvals in a] for a in center_freqs_a]
power_amps_a_m = [[np.mean(allvals) for allvals in a] for a in power_amps_a]
bandwidths_a_m = [[np.mean(allvals) for allvals in a] for a in bandwidths_a]
center_freqs_b_m = [[np.mean(allvals) for allvals in a] for a in center_freqs_b]
power_amps_b_m = [[np.mean(allvals) for allvals in a] for a in power_amps_b]
bandwidths_b_m = [[np.mean(allvals) for allvals in a] for a in bandwidths_b]
center_freqs_max_m = [[np.mean(allvals) for allvals in a] for a in center_freqs_max]
power_amps_max_m = [[np.mean(allvals) for allvals in a] for a in power_amps_max]
bandwidths_max_m = [[np.mean(allvals) for allvals in a] for a in bandwidths_max]
AUC_max_m = [[np.mean(allvals) for allvals in a] for a in AUC_max]
AUC_m = [[np.mean(allvals) for allvals in a] for a in AUC]
AUC_t_m = [[np.mean(allvals) for allvals in a] for a in AUC_t]
AUC_a_m = [[np.mean(allvals) for allvals in a] for a in AUC_a]
AUC_b_m = [[np.mean(allvals) for allvals in a] for a in AUC_b]
AUC_broad_abs_m = [[np.mean(allvals) for allvals in a] for a in AUC_broad_abs]
AUC_t_abs_m = [[np.mean(allvals) for allvals in a] for a in AUC_t_abs]
AUC_a_abs_m = [[np.mean(allvals) for allvals in a] for a in AUC_a_abs]
AUC_b_abs_m = [[np.mean(allvals) for allvals in a] for a in AUC_b_abs]
base_rates_m = [[np.mean(allvals) for allvals in a] for a in base_rates]

offsets_sd = [[np.std(allvals) for allvals in a] for a in offsets]
exponents_sd = [[np.std(allvals) for allvals in a] for a in exponents]
errors_sd = [[np.std(allvals) for allvals in a] for a in errors]
center_freqs_t_sd = [[np.std(allvals) for allvals in a] for a in center_freqs_t]
power_amps_t_sd = [[np.std(allvals) for allvals in a] for a in power_amps_t]
bandwidths_t_sd = [[np.std(allvals) for allvals in a] for a in bandwidths_t]
center_freqs_a_sd = [[np.std(allvals) for allvals in a] for a in center_freqs_a]
power_amps_a_sd = [[np.std(allvals) for allvals in a] for a in power_amps_a]
bandwidths_a_sd = [[np.std(allvals) for allvals in a] for a in bandwidths_a]
center_freqs_b_sd = [[np.std(allvals) for allvals in a] for a in center_freqs_b]
power_amps_b_sd = [[np.std(allvals) for allvals in a] for a in power_amps_b]
bandwidths_b_sd = [[np.std(allvals) for allvals in a] for a in bandwidths_b]
center_freqs_max_sd = [[np.std(allvals) for allvals in a] for a in center_freqs_max]
power_amps_max_sd = [[np.std(allvals) for allvals in a] for a in power_amps_max]
bandwidths_max_sd = [[np.std(allvals) for allvals in a] for a in bandwidths_max]
AUC_max_sd = [[np.std(allvals) for allvals in a] for a in AUC_max]
AUC_sd = [[np.std(allvals) for allvals in a] for a in AUC]
AUC_t_sd = [[np.std(allvals) for allvals in a] for a in AUC_t]
AUC_a_sd = [[np.std(allvals) for allvals in a] for a in AUC_a]
AUC_b_sd = [[np.std(allvals) for allvals in a] for a in AUC_b]
AUC_broad_abs_sd = [[np.std(allvals) for allvals in a] for a in AUC_broad_abs]
AUC_t_abs_sd = [[np.std(allvals) for allvals in a] for a in AUC_t_abs]
AUC_a_abs_sd = [[np.std(allvals) for allvals in a] for a in AUC_a_abs]
AUC_b_abs_sd = [[np.std(allvals) for allvals in a] for a in AUC_b_abs]
base_rates_sd = [[np.std(allvals) for allvals in a] for a in base_rates]

percentiles0 = [25,75] # upper and lower percentiles to compute
offsets_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in offsets]
exponents_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in exponents]
errors_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in errors]
center_freqs_t_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in center_freqs_t]
power_amps_t_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in power_amps_t]
bandwidths_t_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in bandwidths_t]
center_freqs_a_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in center_freqs_a]
power_amps_a_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in power_amps_a]
bandwidths_a_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in bandwidths_a]
center_freqs_b_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in center_freqs_b]
power_amps_b_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in power_amps_b]
bandwidths_b_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in bandwidths_b]
center_freqs_max_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in center_freqs_max]
power_amps_max_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in power_amps_max]
bandwidths_max_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in bandwidths_max]
AUC_max_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in AUC_max]
AUC_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in AUC]
AUC_t_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in AUC_t]
AUC_a_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in AUC_a]
AUC_b_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in AUC_b]
AUC_broad_abs_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in AUC_broad_abs]
AUC_t_abs_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in AUC_t_abs]
AUC_a_abs_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in AUC_a_abs]
AUC_b_abs_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in AUC_b_abs]
base_rates_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in base_rates]

# vs healthy
offsets_tstat0 = [[st.ttest_rel(offsets[0][0],offsets[d][s])[0] if offsets[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
exponents_tstat0 = [[st.ttest_rel(exponents[0][0],exponents[d][s])[0] if exponents[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
errors_tstat0 = [[st.ttest_rel(errors[0][0],errors[d][s])[0] if errors[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_t_tstat0 = [[st.ttest_ind(center_freqs_t[0][0],center_freqs_t[d][s])[0] if center_freqs_t[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_t_tstat0 = [[st.ttest_ind(power_amps_t[0][0],power_amps_t[d][s])[0] if power_amps_t[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_t_tstat0 = [[st.ttest_ind(bandwidths_t[0][0],bandwidths_t[d][s])[0] if bandwidths_t[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_a_tstat0 = [[st.ttest_ind(center_freqs_a[0][0],center_freqs_a[d][s])[0] if center_freqs_a[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_a_tstat0 = [[st.ttest_ind(power_amps_a[0][0],power_amps_a[d][s])[0] if power_amps_a[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_a_tstat0 = [[st.ttest_ind(bandwidths_a[0][0],bandwidths_a[d][s])[0] if bandwidths_a[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_b_tstat0 = [[st.ttest_ind(center_freqs_b[0][0],center_freqs_b[d][s])[0] if center_freqs_b[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_b_tstat0 = [[st.ttest_ind(power_amps_b[0][0],power_amps_b[d][s])[0] if power_amps_b[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_b_tstat0 = [[st.ttest_ind(bandwidths_b[0][0],bandwidths_b[d][s])[0] if bandwidths_b[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_max_tstat0 = [[st.ttest_ind(center_freqs_max[0][0],center_freqs_max[d][s])[0] if center_freqs_max[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_max_tstat0 = [[st.ttest_ind(power_amps_max[0][0],power_amps_max[d][s])[0] if power_amps_max[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_max_tstat0 = [[st.ttest_ind(bandwidths_max[0][0],bandwidths_max[d][s])[0] if bandwidths_max[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_max_tstat0 = [[st.ttest_ind(AUC_max[0][0],AUC_max[d][s])[0] if AUC_max[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_tstat0 = [[st.ttest_rel(AUC[0][0],AUC[d][s])[0] if AUC[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_t_tstat0 = [[st.ttest_rel(AUC_t[0][0],AUC_t[d][s])[0] if AUC_t[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_a_tstat0 = [[st.ttest_rel(AUC_a[0][0],AUC_a[d][s])[0] if AUC_a[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_b_tstat0 = [[st.ttest_rel(AUC_b[0][0],AUC_b[d][s])[0] if AUC_b[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_broad_abs_tstat0 = [[st.ttest_rel(AUC_broad_abs[0][0],AUC_broad_abs[d][s])[0] if AUC_broad_abs[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_t_abs_tstat0 = [[st.ttest_rel(AUC_t_abs[0][0],AUC_t_abs[d][s])[0] if AUC_t_abs[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_a_abs_tstat0 = [[st.ttest_rel(AUC_a_abs[0][0],AUC_a_abs[d][s])[0] if AUC_a_abs[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_b_abs_tstat0 = [[st.ttest_rel(AUC_b_abs[0][0],AUC_b_abs[d][s])[0] if AUC_b_abs[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
base_rates_tstat0 = [[st.ttest_rel(base_rates[0][0],base_rates[d][s])[0] if base_rates[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]

offsets_p0 = [[st.ttest_rel(offsets[0][0],offsets[d][s])[1] if offsets[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
exponents_p0 = [[st.ttest_rel(exponents[0][0],exponents[d][s])[1] if exponents[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
errors_p0 = [[st.ttest_rel(errors[0][0],errors[d][s])[1] if errors[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_t_p0 = [[st.ttest_ind(center_freqs_t[0][0],center_freqs_t[d][s])[1] if center_freqs_t[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_t_p0 = [[st.ttest_ind(power_amps_t[0][0],power_amps_t[d][s])[1] if power_amps_t[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_t_p0 = [[st.ttest_ind(bandwidths_t[0][0],bandwidths_t[d][s])[1] if bandwidths_t[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_a_p0 = [[st.ttest_ind(center_freqs_a[0][0],center_freqs_a[d][s])[1] if center_freqs_a[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_a_p0 = [[st.ttest_ind(power_amps_a[0][0],power_amps_a[d][s])[1] if power_amps_a[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_a_p0 = [[st.ttest_ind(bandwidths_a[0][0],bandwidths_a[d][s])[1] if bandwidths_a[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_b_p0 = [[st.ttest_ind(center_freqs_b[0][0],center_freqs_b[d][s])[1] if center_freqs_b[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_b_p0 = [[st.ttest_ind(power_amps_b[0][0],power_amps_b[d][s])[1] if power_amps_b[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_b_p0 = [[st.ttest_ind(bandwidths_b[0][0],bandwidths_b[d][s])[1] if bandwidths_b[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_max_p0 = [[st.ttest_ind(center_freqs_max[0][0],center_freqs_max[d][s])[1] if center_freqs_max[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_max_p0 = [[st.ttest_ind(power_amps_max[0][0],power_amps_max[d][s])[1] if power_amps_max[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_max_p0 = [[st.ttest_ind(bandwidths_max[0][0],bandwidths_max[d][s])[1] if bandwidths_max[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_max_p0 = [[st.ttest_ind(AUC_max[0][0],AUC_max[d][s])[1] if AUC_max[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_p0 = [[st.ttest_rel(AUC[0][0],AUC[d][s])[1] if AUC[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_t_p0 = [[st.ttest_rel(AUC_t[0][0],AUC_t[d][s])[1] if AUC_t[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_a_p0 = [[st.ttest_rel(AUC_a[0][0],AUC_a[d][s])[1] if AUC_a[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_b_p0 = [[st.ttest_rel(AUC_b[0][0],AUC_b[d][s])[1] if AUC_b[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_broad_abs_p0 = [[st.ttest_rel(AUC_broad_abs[0][0],AUC_broad_abs[d][s])[1] if AUC_broad_abs[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_t_abs_p0 = [[st.ttest_rel(AUC_t_abs[0][0],AUC_t_abs[d][s])[1] if AUC_t_abs[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_a_abs_p0 = [[st.ttest_rel(AUC_a_abs[0][0],AUC_a_abs[d][s])[1] if AUC_a_abs[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_b_abs_p0 = [[st.ttest_rel(AUC_b_abs[0][0],AUC_b_abs[d][s])[1] if AUC_b_abs[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
base_rates_p0 = [[st.ttest_rel(base_rates[0][0],base_rates[d][s])[1] if base_rates[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]

offsets_cd0 = [[cohen_d(offsets[0][0],offsets[d][s]) if offsets[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
exponents_cd0 = [[cohen_d(exponents[0][0],exponents[d][s]) if exponents[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
errors_cd0 = [[cohen_d(errors[0][0],errors[d][s]) if errors[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_t_cd0 = [[cohen_d(center_freqs_t[0][0],center_freqs_t[d][s]) if center_freqs_t[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_t_cd0 = [[cohen_d(power_amps_t[0][0],power_amps_t[d][s]) if power_amps_t[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_t_cd0 = [[cohen_d(bandwidths_t[0][0],bandwidths_t[d][s]) if bandwidths_t[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_a_cd0 = [[cohen_d(center_freqs_a[0][0],center_freqs_a[d][s]) if center_freqs_a[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_a_cd0 = [[cohen_d(power_amps_a[0][0],power_amps_a[d][s]) if power_amps_a[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_a_cd0 = [[cohen_d(bandwidths_a[0][0],bandwidths_a[d][s]) if bandwidths_a[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_b_cd0 = [[cohen_d(center_freqs_b[0][0],center_freqs_b[d][s]) if center_freqs_b[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_b_cd0 = [[cohen_d(power_amps_b[0][0],power_amps_b[d][s]) if power_amps_b[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_b_cd0 = [[cohen_d(bandwidths_b[0][0],bandwidths_b[d][s]) if bandwidths_b[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_max_cd0 = [[cohen_d(center_freqs_max[0][0],center_freqs_max[d][s]) if center_freqs_max[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_max_cd0 = [[cohen_d(power_amps_max[0][0],power_amps_max[d][s]) if power_amps_max[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_max_cd0 = [[cohen_d(bandwidths_max[0][0],bandwidths_max[d][s]) if bandwidths_max[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_max_cd0 = [[cohen_d(AUC_max[0][0],AUC_max[d][s]) if AUC_max[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_cd0 = [[cohen_d(AUC[0][0],AUC[d][s]) if AUC[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_t_cd0 = [[cohen_d(AUC_t[0][0],AUC_t[d][s]) if AUC_t[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_a_cd0 = [[cohen_d(AUC_a[0][0],AUC_a[d][s]) if AUC_a[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_b_cd0 = [[cohen_d(AUC_b[0][0],AUC_b[d][s]) if AUC_b[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_broad_abs_cd0 = [[cohen_d(AUC_broad_abs[0][0],AUC_broad_abs[d][s]) if AUC_broad_abs[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_t_abs_cd0 = [[cohen_d(AUC_t_abs[0][0],AUC_t_abs[d][s]) if AUC_t_abs[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_a_abs_cd0 = [[cohen_d(AUC_a_abs[0][0],AUC_a_abs[d][s]) if AUC_a_abs[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_b_abs_cd0 = [[cohen_d(AUC_b_abs[0][0],AUC_b_abs[d][s]) if AUC_b_abs[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
base_rates_cd0 = [[cohen_d(base_rates[0][0],base_rates[d][s]) if base_rates[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]

df = pd.DataFrame(columns=["Metric",
			"Condition",
			"Mean",
			"SD",
			"t-stat",
			"p-value",
			"Cohen's d"])


metric_names = ["Base Rates",
				"Offset",
				"Exponent",
				"Error",
				"PSD Alpha+Beta AUC",
				"PSD Theta AUC",
				"PSD Alpha AUC",
				"PSD Beta AUC",
				"Aperiodic AUC",
				"Periodic Theta AUC",
				"Periodic Alpha AUC",
				"Periodic Beta AUC",
				"Max Center Frequency",
				"Max Relative Power",
				"Max Bandwidth",
				"Max Peak AUC"]

m = [base_rates_m,
	offsets_m,
	exponents_m,
	errors_m,
	AUC_broad_abs_m,
	AUC_t_abs_m,
	AUC_a_abs_m,
	AUC_b_abs_m,
	AUC_m,
	AUC_t_m,
	AUC_a_m,
	AUC_b_m,
	center_freqs_max_m,
	power_amps_max_m,
	bandwidths_max_m,
	AUC_max_m]
sd = [base_rates_sd,
	offsets_sd,
	exponents_sd,
	errors_sd,
	AUC_broad_abs_sd,
	AUC_t_abs_sd,
	AUC_a_abs_sd,
	AUC_b_abs_sd,
	AUC_sd,
	AUC_t_sd,
	AUC_a_sd,
	AUC_b_sd,
	center_freqs_max_sd,
	power_amps_max_sd,
	bandwidths_max_sd,
	AUC_max_sd]

tstat0 = [base_rates_tstat0,
	offsets_tstat0,
	exponents_tstat0,
	errors_tstat0,
	AUC_broad_abs_tstat0,
	AUC_t_abs_tstat0,
	AUC_a_abs_tstat0,
	AUC_b_abs_tstat0,
	AUC_tstat0,
	AUC_t_tstat0,
	AUC_a_tstat0,
	AUC_b_tstat0,
	center_freqs_max_tstat0,
	power_amps_max_tstat0,
	bandwidths_max_tstat0,
	AUC_max_tstat0]
p0 = [base_rates_p0,
	offsets_p0,
	exponents_p0,
	errors_p0,
	AUC_broad_abs_p0,
	AUC_t_abs_p0,
	AUC_a_abs_p0,
	AUC_b_abs_p0,
	AUC_p0,
	AUC_t_p0,
	AUC_a_p0,
	AUC_b_p0,
	center_freqs_max_p0,
	power_amps_max_p0,
	bandwidths_max_p0,
	AUC_max_p0]
cd0 = [base_rates_cd0,
	offsets_cd0,
	exponents_cd0,
	errors_cd0,
	AUC_broad_abs_cd0,
	AUC_t_abs_cd0,
	AUC_a_abs_cd0,
	AUC_b_abs_cd0,
	AUC_cd0,
	AUC_t_cd0,
	AUC_a_cd0,
	AUC_b_cd0,
	center_freqs_max_cd0,
	power_amps_max_cd0,
	bandwidths_max_cd0,
	AUC_max_cd0]

# vs Healthy
for lind1,labeli1 in enumerate(x_labels2):
	for lind2,labeli2 in enumerate(labeli1):
		for ind1,metric in enumerate(metric_names):
			df = df.append({"Metric":metric,
						"Condition":labeli2,
						"Mean":m[ind1][lind1][lind2],
						"SD":sd[ind1][lind1][lind2],
						"t-stat":tstat0[ind1][lind1][lind2],
						"p-value":p0[ind1][lind1][lind2],
						"Cohen's d":cd0[ind1][lind1][lind2]},
						ignore_index = True)

df.to_csv('figs_PSDvsSpikes_V3/stats_PSD.csv')

figsize1 = (7,5)
figsize2 = (14,5)
dh1 = 0.03

p_thresh = 0.05
c_thresh = 1.

metrics = [base_rates,
			offsets,
			exponents,
			errors,
			AUC_broad_abs,
			AUC_t_abs,
			AUC_a_abs,
			AUC_b_abs,
			AUC,
			AUC_t,
			AUC_a,
			AUC_b,
			center_freqs_max,
			power_amps_max,
			bandwidths_max,
			AUC_max]

base_rates_m = [[np.mean(allvals) for allvals in a] for a in base_rates]

def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    if ax is None:
        ax = plt.gca()
    
    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="k", alpha = 0.4)
	
    return ax

# Goal: Find which metric correlates best with base rates
def run_correlations(data,ylabel_plot,filename):
	d1 = data[0]
	for dind,d2 in enumerate(data[1:]):
		# Flatten lists across conditions and remove nans
		d1_flat0 = []
		d2_flat0 = []
		for i in d1:
			for j in i:
				for k in j:
					d1_flat0.append(k)
		for i in d2:
			for j in i:
				for k in j:
					d2_flat0.append(k)
		
		d1_flat = [x for x,y in zip(d1_flat0,d2_flat0) if ((str(x) != 'nan') and (str(y) != 'nan'))]
		d2_flat = [y for x,y in zip(d1_flat0,d2_flat0) if ((str(x) != 'nan') and (str(y) != 'nan'))]
		
		# Run correlation using data across all conditions
		r,p = st.pearsonr(d1_flat,d2_flat)
		
		# Add regression line + 95% confidence interval
		m, b = np.polyfit(d1_flat, d2_flat, 1)
		y_model = [m*x+b for x in d1_flat]
		n0 = len(d2_flat)
		m0 = 2.
		dof = n0 - m0
		t = st.t.ppf(0.975, n0 - m0)
		resid = [y - yp for y,yp in zip(d2_flat,y_model)]
		chi2 = [np.sum((r / yp)**2) for r,yp in zip(resid,y_model)]
		chi2_red = [c / dof for c in chi2]
		s_err = np.sqrt(np.sum([r**2 for r in resid]) / dof)
		x_highres = np.linspace(np.min(d1_flat), np.max(d1_flat), 100)
		y_highres = [m*x+b for x in x_highres]
		
		# Plot Scatter
		fig, ax = plt.subplots(figsize=(7, 5))
		for cind2,s in enumerate(severities):
			for cind1 in range(len(doses)):
				d1_nonan = [x for x,y in zip(d1[cind1][cind2],d2[cind1][cind2]) if ((str(x) != 'nan') and (str(y) != 'nan'))]
				d2_nonan = [y for x,y in zip(d1[cind1][cind2],d2[cind1][cind2]) if ((str(x) != 'nan') and (str(y) != 'nan'))]
				ax.scatter(d1_nonan,d2_nonan,s=8**2,c=colors_conds[cind1][cind2].reshape(1,-1))
		ax.plot(x_highres, y_highres, c='k')
		plot_ci_manual(t, s_err, n0, d1_flat, x_highres, y_highres, ax=ax)
		
		ax.set_xlabel('Spike Rate (Hz)')
		ax.set_ylabel(ylabel_plot[dind+1])
		ax.set_title('R = ' + f"{r:0.2f}", loc='right', fontsize = 24)
		fig.tight_layout()
		fig.savefig('figs_PSDvsSpikes_V3/base_rate_vs_'+filename[dind+1]+'.png',dpi=300,transparent=True)
		plt.close(fig)

run_correlations(metrics,metric_names,[m.replace(" ","") for m in metric_names])


cmap = plt.cm.get_cmap('hot')#.reversed()
norm = plt.Normalize(vmin=severities[0], vmax=severities[-1]+0.26)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

colors_conds_simple = ['gray','tab:purple','tab:purple','tab:purple','tab:purple']

def run_correlations_zerodose(data,ylabel_plot,filename,mode):
	d1 = data[0]
	for dind,d2 in enumerate(data[1:]):
		# Flatten lists across conditions and remove nans
		d1_flat0 = []
		d2_flat0 = []
		for i in d1:
			for k in i:
				d1_flat0.append(k)
		for i in d2:
			for k in i:
				d2_flat0.append(k)
		
		d1_flat = [x for x,y in zip(d1_flat0,d2_flat0) if ((str(x) != 'nan') and (str(y) != 'nan'))]
		d2_flat = [y for x,y in zip(d1_flat0,d2_flat0) if ((str(x) != 'nan') and (str(y) != 'nan'))]
		
		# Run correlation using data across all conditions
		r,p = st.pearsonr(d1_flat,d2_flat)
		
		# Add regression line + 95% confidence interval
		m, b = np.polyfit(d1_flat, d2_flat, 1)
		y_model = [m*x+b for x in d1_flat]
		n0 = len(d2_flat)
		m0 = 2.
		dof = n0 - m0
		t = st.t.ppf(0.975, n0 - m0)
		resid = [y - yp for y,yp in zip(d2_flat,y_model)]
		chi2 = [np.sum((r / yp)**2) for r,yp in zip(resid,y_model)]
		chi2_red = [c / dof for c in chi2]
		s_err = np.sqrt(np.sum([r**2 for r in resid]) / dof)
		x_highres = np.linspace(np.min(d1_flat), np.max(d1_flat), 100)
		y_highres = [m*x+b for x in x_highres]
		
		# Plot Scatter
		#fig, ax = plt.subplots(figsize=(7, 5))
		fig, ax = plt.subplots(figsize=(4.4, 5.3))
		for cind2,s in enumerate(severities):
			if mode == 1: col = colors_conds[0][cind2].reshape(1,-1)
			elif mode == 2: col = sm.to_rgba(severities[cind2])
			elif mode == 3: col = colors_conds_simple[cind2]
			d1_nonan = [x for x,y in zip(d1[cind2],d2[cind2]) if ((str(x) != 'nan') and (str(y) != 'nan'))]
			d2_nonan = [y for x,y in zip(d1[cind2],d2[cind2]) if ((str(x) != 'nan') and (str(y) != 'nan'))]
			ax.scatter(d1_nonan,d2_nonan,s=8**2,c=col)
		ax.plot(x_highres, y_highres, c='k')
		plot_ci_manual(t, s_err, n0, d1_flat, x_highres, y_highres, ax=ax)
		
		ax.set_xlabel('Spike Rate (Hz)')
		if ylabel_plot[dind+1] == 'Aperiodic AUC':
			ax.set_ylabel('1/f')
		elif ylabel_plot[dind+1] == 'PSD Alpha AUC':
			ax.set_ylabel(r'$\alpha$')
		elif ylabel_plot[dind+1] == 'PSD Theta AUC':
			ax.set_ylabel(r'$\theta$')
		else:
			ax.set_ylabel(ylabel_plot[dind+1])
		ax.set_title('R=' + f"{r:0.2f}", loc='right', fontsize = 24)
		fig.tight_layout()
		fig.savefig('figs_PSDvsSpikes_V3/base_rate_vs_'+filename[dind+1]+'.png',dpi=300,transparent=True)
		plt.close(fig)

metrics_zerodose = [met[0] for met in metrics]
run_correlations_zerodose(metrics_zerodose,metric_names,[m.replace(" ","")+'_zerodose' for m in metric_names],1)
run_correlations_zerodose(metrics_zerodose,metric_names,[m.replace(" ","")+'_zerodose_hot' for m in metric_names],2)
run_correlations_zerodose(metrics_zerodose,metric_names,[m.replace(" ","")+'_zerodose_simple' for m in metric_names],3)

def run_correlations_inset(data,ylabel_plot,filename):
	data0 = [met[0] for met in data]
	
	d1 = data[0]
	d10 = data0[0]
	for dind,(d2,d20) in enumerate(zip(data[1:],data0[1:])):
		
		# Flatten lists across conditions and remove nans
		d1_flat0 = []
		d2_flat0 = []
		for i in d1:
			for j in i:
				for k in j:
					d1_flat0.append(k)
		for i in d2:
			for j in i:
				for k in j:
					d2_flat0.append(k)
		d1_flat00 = []
		d2_flat00 = []
		for i in d10:
			for k in i:
				d1_flat00.append(k)
		for i in d20:
			for k in i:
				d2_flat00.append(k)
		
		d1_flat = [x for x,y in zip(d1_flat0,d2_flat0) if ((str(x) != 'nan') and (str(y) != 'nan'))]
		d2_flat = [y for x,y in zip(d1_flat0,d2_flat0) if ((str(x) != 'nan') and (str(y) != 'nan'))]
		d1_flat01 = [x for x,y in zip(d1_flat00,d2_flat00) if ((str(x) != 'nan') and (str(y) != 'nan'))]
		d2_flat01 = [y for x,y in zip(d1_flat00,d2_flat00) if ((str(x) != 'nan') and (str(y) != 'nan'))]
		
		# Run correlation using data across all conditions
		r,p = st.pearsonr(d1_flat,d2_flat)
		r0,p0 = st.pearsonr(d1_flat01,d2_flat01)
		
		# Add regression line + 95% confidence interval
		m, b = np.polyfit(d1_flat, d2_flat, 1)
		m01, b01 = np.polyfit(d1_flat01, d2_flat01, 1)
		y_model = [m*x+b for x in d1_flat]
		y_model0 = [m01*x+b01 for x in d1_flat01]
		n0 = len(d2_flat)
		n00 = len(d2_flat01)
		m0 = 2.
		m00 = 2.
		dof = n0 - m0
		dof0 = n00 - m00
		t = st.t.ppf(0.975, n0 - m0)
		t0 = st.t.ppf(0.975, n00 - m00)
		resid = [y - yp for y,yp in zip(d2_flat,y_model)]
		resid0 = [y - yp for y,yp in zip(d2_flat01,y_model0)]
		chi2 = [np.sum((rt / yp)**2) for rt,yp in zip(resid,y_model)]
		chi20 = [np.sum((rt / yp)**2) for rt,yp in zip(resid0,y_model0)]
		chi2_red = [c / dof for c in chi2]
		chi2_red0 = [c / dof0 for c in chi20]
		s_err = np.sqrt(np.sum([re**2 for re in resid]) / dof)
		s_err0 = np.sqrt(np.sum([re**2 for re in resid0]) / dof0)
		x_highres = np.linspace(np.min(d1_flat), np.max(d1_flat), 100)
		x_highres0 = np.linspace(np.min(d1_flat01), np.max(d1_flat01), 100)
		y_highres = [m*x+b for x in x_highres]
		y_highres0 = [m01*x+b01 for x in x_highres0]

		# Plot Scatter
		fig, ax = plt.subplots(figsize=(7, 5))
		if ylabel_plot[dind+1] == 'Aperiodic AUC':
			inset = ax.inset_axes([.06,.59,.4,.3])
		else:
			inset = ax.inset_axes([.12,.59,.4,.3])
		
		for cind2,s in enumerate(severities):
			d1_nonan0 = [x for x,y in zip(d10[cind2],d20[cind2]) if ((str(x) != 'nan') and (str(y) != 'nan'))]
			d2_nonan0 = [y for x,y in zip(d10[cind2],d20[cind2]) if ((str(x) != 'nan') and (str(y) != 'nan'))]
			ax.scatter(d1_nonan0,d2_nonan0,s=8**2,c=colors_conds[0][cind2].reshape(1,-1))
		for cind2,s in enumerate(severities):
			for cind1 in range(len(doses)):
				d1_nonan = [x for x,y in zip(d1[cind1][cind2],d2[cind1][cind2]) if ((str(x) != 'nan') and (str(y) != 'nan'))]
				d2_nonan = [y for x,y in zip(d1[cind1][cind2],d2[cind1][cind2]) if ((str(x) != 'nan') and (str(y) != 'nan'))]
				inset.scatter(d1_nonan,d2_nonan,s=8**2,c=colors_conds[cind1][cind2].reshape(1,-1))
		ax.plot(x_highres0, y_highres0, c='k')
		inset.plot(x_highres, y_highres, c='k')
		plot_ci_manual(t0, s_err0, n00, d1_flat01, x_highres0, y_highres0, ax=ax)
		plot_ci_manual(t, s_err, n0, d1_flat, x_highres, y_highres, ax=inset)
		
		ax.set_xlabel('Spike Rate (Hz)')
		if ylabel_plot[dind+1] == 'Aperiodic AUC':
			ax.set_ylabel('1/f')
		elif ylabel_plot[dind+1] == 'PSD Alpha AUC':
			ax.set_ylabel(r'$\alpha$')
		elif ylabel_plot[dind+1] == 'PSD Theta AUC':
			ax.set_ylabel(r'$\theta$')
		else:
			ax.set_ylabel(ylabel_plot[dind+1])
		inset.text(0.02,0.75,f"{r:0.2f}", horizontalalignment='left', verticalalignment='center', fontsize = 24, transform = inset.transAxes)
		ax.set_title('R = ' + f"{r0:0.2f}", loc='right', fontsize = 24)
		fig.tight_layout()
		fig.savefig('figs_PSDvsSpikes_V3/base_rate_vs_'+filename[dind+1]+'.png',dpi=300,transparent=True)
		plt.close(fig)
		print(ylabel_plot[dind+1])
		print('MDD Only; r = '+str(r0)+'; p = '+str(p0))
		print('With Doses; r = '+str(r)+'; p = '+str(p))

run_correlations_inset(metrics,metric_names,[m.replace(" ","")+'_inset' for m in metric_names])

# Run Association plots
# x axis = Spike Rate
# y axis = PSD metric
# color axis = Simulated Dose (hot)
# highlight zero dose in magenta outline or something
# Grey box = healthy range

def run_associations(data,data_m,data_sd,ylabel_plot,filename):
	cmap_to_use = plt.cm.get_cmap('hot').reversed()
	
	healthy_spike_range = [np.min(data[0][0][0]),np.max(data[0][0][0])]
	healthy_psd_range_y1 = [[np.min(d[0][0]),np.min(d[0][0])] for d in data[1:]]
	healthy_psd_range_y2 = [[np.max(d[0][0]),np.max(d[0][0])] for d in data[1:]]
	spike_metric_m = data_m[0]
	psd_metrics_m = data_m[1:]
	spike_metric_sd = data_sd[0]
	psd_metrics_sd = data_sd[1:]
	
	for indm, psd_m in enumerate(psd_metrics_m):
		for cind2,s in enumerate(severities):
			fig, ax = plt.subplots(figsize=(7, 5))
			if cind2 == 0: # skip healthy
				continue
			x = [d[cind2] for d in spike_metric_m] # vector of doses for a given severity
			y = [d[cind2] for d in psd_metrics_m[indm]] # vector of doses for a given severity
			x_e = [d[cind2] for d in spike_metric_sd] # vector of doses for a given severity
			y_e = [d[cind2] for d in psd_metrics_sd[indm]] # vector of doses for a given severity
			
			ax.fill_between(healthy_spike_range,y1=healthy_psd_range_y1[indm],y2=healthy_psd_range_y2[indm],color='gray',alpha=0.4)
			ax.errorbar(x,y,xerr=x_e,yerr=y_e,ecolor='k', elinewidth=3, capsize=5,capthick=3,linestyle='',zorder=1)
			ax.scatter(x[0],y[0],c='magenta', s=18**2, edgecolors='magenta')
			img = ax.scatter(x,y,c=doses,cmap=cmap_to_use, s=14**2, edgecolors='black', vmin=doses[0], vmax=doses[-1])
			xlims = ax.get_xlim()
			ylims = ax.get_ylim()
			ax.plot([healthy_spike_range[0],healthy_spike_range[0]],ylims,c='gray',ls='dashed')
			ax.plot([healthy_spike_range[1],healthy_spike_range[1]],ylims,c='gray',ls='dashed')
			ax.plot(xlims,healthy_psd_range_y1[indm],c='gray',ls='dashed')
			ax.plot(xlims,healthy_psd_range_y2[indm],c='gray',ls='dashed')
			ax.set_xlim(xlims)
			ax.set_ylim(ylims)

			ax.set_xlabel('Spike Rate (Hz)')
			ax.set_ylabel(ylabel_plot[indm+1])
			fig.colorbar(img,label='Simulated Dose', ticks=doses)
			fig.tight_layout()
			fig.savefig('figs_PSDvsSpikes_V3/base_rate_vs_'+filename[indm+1]+'_association_severity'+str(s)+'.png',dpi=300,transparent=True)
			plt.close(fig)

run_associations(metrics,m,sd,metric_names,[m.replace(" ","") for m in metric_names])
