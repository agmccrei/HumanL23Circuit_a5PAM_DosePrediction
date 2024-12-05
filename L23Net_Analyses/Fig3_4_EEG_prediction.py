################################################################################
# Import, Set up MPI Variables, Load Necessary Files
################################################################################
import os
from os.path import join
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.ticker import (LogLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.markers import MarkerStyle
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
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import shap

np.random.seed(seed=1234)

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':26})

NPermutes = 50
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

doses_of_interest = np.array([0.50,0.75,1.25]) # Should be of length severities - 1 - defunct

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
				continue
			
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
			
			#fm.plot(save_fig=True,file_name='figs_FOOOFfits_V1_nperseg_80000/fits_'+conds[cind1][cind2]+'_'+str(seed)+'.png')
			#plt.close()
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

# Z-score and normalize data in log10 space
def normalize_results(input_data):
	data1 = np.log10(input_data) # log10
	healthy_m = np.mean(data1[0][0])
	healthy_sd = np.std(data1[0][0])
	data2 = np.array(data1).reshape(-1,1) # reshape for scaling, Scalers require this format
	#z_data = [(d - healthy_m)/healthy_sd for d in data2]
	z_data = StandardScaler().fit_transform(data2) # z-score scaling
	scaled_data = MinMaxScaler((-1,1)).fit_transform(z_data).flatten() # scaled between -1, 1
	output_data = np.array(scaled_data).reshape(np.shape(data1)) # retrieve original shape of data
	return output_data

offsets = normalize_results(offsets)
exponents = normalize_results(exponents)
errors = normalize_results(errors)
center_freqs_max = normalize_results(center_freqs_max)
power_amps_max = normalize_results(power_amps_max)
bandwidths_max = normalize_results(bandwidths_max)
AUC_max = normalize_results(AUC_max)
AUC = normalize_results(AUC)
AUC_t = normalize_results(AUC_t)
AUC_a = normalize_results(AUC_a)
AUC_b = normalize_results(AUC_b)
AUC_broad_abs = normalize_results(AUC_broad_abs)
AUC_t_abs = normalize_results(AUC_t_abs)
AUC_a_abs = normalize_results(AUC_a_abs)
AUC_b_abs = normalize_results(AUC_b_abs)

# Area under frequency bands - bar plots
offsets_m = [[np.mean(allvals) for allvals in a] for a in offsets]
exponents_m = [[np.mean(allvals) for allvals in a] for a in exponents]
errors_m = [[np.mean(allvals) for allvals in a] for a in errors]
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

offsets_sd = [[np.std(allvals) for allvals in a] for a in offsets]
exponents_sd = [[np.std(allvals) for allvals in a] for a in exponents]
errors_sd = [[np.std(allvals) for allvals in a] for a in errors]
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

percentiles0 = [0,100] # upper and lower percentiles to compute
offsets_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in offsets]
exponents_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in exponents]
errors_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in errors]
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

# vs healthy
offsets_tstat0 = [[st.ttest_rel(offsets[0][0],offsets[d][s])[0] if offsets[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
exponents_tstat0 = [[st.ttest_rel(exponents[0][0],exponents[d][s])[0] if exponents[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
errors_tstat0 = [[st.ttest_rel(errors[0][0],errors[d][s])[0] if errors[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
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

offsets_p0 = [[st.ttest_rel(offsets[0][0],offsets[d][s])[1] if offsets[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
exponents_p0 = [[st.ttest_rel(exponents[0][0],exponents[d][s])[1] if exponents[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
errors_p0 = [[st.ttest_rel(errors[0][0],errors[d][s])[1] if errors[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
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

offsets_cd0 = [[cohen_d(offsets[0][0],offsets[d][s]) if offsets[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
exponents_cd0 = [[cohen_d(exponents[0][0],exponents[d][s]) if exponents[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
errors_cd0 = [[cohen_d(errors[0][0],errors[d][s]) if errors[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
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

df = pd.DataFrame(columns=["Metric",
			"Condition",
			"Mean",
			"SD",
			"t-stat",
			"p-value",
			"Cohen's d"])


metric_names = ["Offset",
				"Exponent",
				"Error",
				"PSD AUC",
				"PSD Theta AUC",
				"PSD Alpha AUC",
				"PSD Beta AUC",
				"Aperiodic AUC",
				"Periodic Theta AUC",
				"Periodic Alpha AUC",
				"Periodic Beta AUC",
				"Center Frequency (max)",
				"Relative Power (max)",
				"Bandwidth (max)",
				"AUC (max peak)"]

m = [offsets_m,
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
sd = [offsets_sd,
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

tstat0 = [offsets_tstat0,
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
p0 = [offsets_p0,
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
cd0 = [offsets_cd0,
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

df.to_csv('figs_ManualLinearRegression_V13/stats_PSD.csv')

# Fit functions to dose-curves
def linear(x, slope, intercept):
	return slope*x + intercept

def multi_linear(x, a, b, c, d):
    return a + b*x[0] + c*x[1] + d*x[2]

fig_dosefits, ax_dosefits = plt.subplots(1, 4, sharey=True, figsize=(16,4))
def run_multi_line_fits(data,opt_doses):
	# General Equation:
	# Optimal Dose = a + b*aperiodic_AUC + c*theta_AUC + d*alpha_AUC
	
	# Initial parameter estimates
	data = np.array(data)
	s1 = (np.max(opt_doses)-np.min(opt_doses))/(np.mean(data[0][(opt_doses==np.max(opt_doses))])-np.mean(data[0][(opt_doses==np.min(opt_doses))]))
	s2 = (np.max(opt_doses)-np.min(opt_doses))/(np.mean(data[1][(opt_doses==np.max(opt_doses))])-np.mean(data[1][(opt_doses==np.min(opt_doses))]))
	s3 = (np.max(opt_doses)-np.min(opt_doses))/(np.mean(data[2][(opt_doses==np.max(opt_doses))])-np.mean(data[2][(opt_doses==np.min(opt_doses))]))
	s4 = (np.max(opt_doses)-np.min(opt_doses))/(np.mean(data[3][(opt_doses==np.max(opt_doses))])-np.mean(data[3][(opt_doses==np.min(opt_doses))]))
	
	i0 = np.mean([od - s1*d1 - s2*d2 - s3*d3 for od,d1,d2,d3 in zip(opt_doses,data[0],data[1],data[2])])
	i1 = np.mean([od - s1*d1 for od,d1 in zip(opt_doses,data[0])])
	i2 = np.mean([od - s2*d2 for od,d2 in zip(opt_doses,data[1])])
	i3 = np.mean([od - s3*d3 for od,d3 in zip(opt_doses,data[2])])
	i4 = np.mean([od - s4*d4 for od,d4 in zip(opt_doses,data[3])])
	
	p0_l = [i0, s1, s2, s3]
	p1 = [s1, i1]
	p2 = [s2, i2]
	p3 = [s3, i3]
	p4 = [s4, i4]
	
	coeff_l = curve_fit(multi_linear, data[:3], opt_doses, p0 = p0_l, full_output=True)
	coeff_l_1 = curve_fit(linear, data[0], opt_doses, p0 = p1, full_output=True)
	coeff_l_2 = curve_fit(linear, data[1], opt_doses, p0 = p2, full_output=True)
	coeff_l_3 = curve_fit(linear, data[2], opt_doses, p0 = p3, full_output=True)
	coeff_l_4 = curve_fit(linear, data[3], opt_doses, p0 = p4, full_output=True)
	
	xvals_highres_1 = np.linspace(np.min(inputs_doses[0]),np.max(inputs_doses[0]),10**3) # use full dataset for plotting full range for each line
	xvals_highres_2 = np.linspace(np.min(inputs_doses[1]),np.max(inputs_doses[1]),10**3)
	xvals_highres_3 = np.linspace(np.min(inputs_doses[2]),np.max(inputs_doses[2]),10**3)
	xvals_highres_4 = np.linspace(np.min(inputs_doses[3]),np.max(inputs_doses[3]),10**3)
	
	l_fit1 = linear(xvals_highres_1, *coeff_l_1[0])
	l_fit2 = linear(xvals_highres_2, *coeff_l_2[0])
	l_fit3 = linear(xvals_highres_3, *coeff_l_3[0])
	l_fit4 = linear(xvals_highres_4, *coeff_l_4[0])
	
	ax_dosefits[0].scatter(data[0], opt_doses, c='k')
	ax_dosefits[0].plot(xvals_highres_1, l_fit1, c='tab:orange', lw=4,alpha=0.2)
	ax_dosefits[0].set_xlabel('AUC')
	ax_dosefits[1].scatter(data[1], opt_doses, c='k')
	ax_dosefits[1].plot(xvals_highres_2, l_fit2, c='tab:green', lw=4,alpha=0.2)
	ax_dosefits[1].set_xlabel(r'$\theta$')
	ax_dosefits[2].scatter(data[2], opt_doses, c='k')
	ax_dosefits[2].plot(xvals_highres_3, l_fit3, c='tab:blue', lw=4,alpha=0.2)
	ax_dosefits[2].set_xlabel(r'$\alpha$')
	ax_dosefits[3].scatter(data[3], opt_doses, c='k')
	ax_dosefits[3].plot(xvals_highres_4, l_fit4, c='tab:purple', lw=4,alpha=0.2)
	ax_dosefits[3].set_xlabel(r'$\alpha$' + '+' + r'$\beta$')
	ax_dosefits[0].set_ylabel('Optimal Dose')
	
	return coeff_l, fig_dosefits, coeff_l_1, coeff_l_2, coeff_l_3, coeff_l_4

# Find optimal doses for each individual
def find_optimal_dose_and_range(all_data,mh,lh,uh):
	mmdr = []
	mldr = []
	mudr = []
	for sind,s in enumerate(severities):
		for i in range(0,N_seeds):
			
			# Find optimal dose using multivariate fits
			data = np.transpose(np.array(all_data), axes = [0, 2, 3, 1])
			s1 = (doses[-1]-doses[0])/(np.mean(data[0][sind][i][-1])-np.mean(data[0][sind][i][0]))
			s2 = (doses[-1]-doses[0])/(np.mean(data[1][sind][i][-1])-np.mean(data[1][sind][i][0]))
			s3 = (doses[-1]-doses[0])/(np.mean(data[2][sind][i][-1])-np.mean(data[2][sind][i][0]))
			i0 = np.mean([od - s1*d1 - s2*d2 - s3*d3 for od,d1,d2,d3 in zip(doses,data[0][sind][i],data[1][sind][i],data[2][sind][i])])
			p0_l = [i0, s1, s2, s3]
			
			coeff_d = curve_fit(multi_linear, [tempd[sind][i] for tempd in data], doses, p0 = p0_l, full_output=True)
			
			MV_middle_dose_range = multi_linear(np.array(mh), *coeff_d[0])
			MV_lower_dose_range = multi_linear(np.array(uh), *coeff_d[0])
			MV_upper_dose_range = multi_linear(np.array(lh), *coeff_d[0])
			
			xvals_highres_1 = np.linspace(np.nanmin(data[0]),np.nanmax(data[0]),5**2) # use full dataset for plotting full range for each line
			xvals_highres_2 = np.linspace(np.nanmin(data[1]),np.nanmax(data[1]),5**2)
			xvals_highres_3 = np.linspace(np.nanmin(data[2]),np.nanmax(data[2]),5**2)
			xvals_h1 = np.array([])
			xvals_h2 = np.array([])
			xvals_h3 = np.array([])
			for x1 in xvals_highres_1:
				for x2 in xvals_highres_2:
					for x3 in xvals_highres_3:
						xvals_h1 = np.append(xvals_h1,x1)
						xvals_h2 = np.append(xvals_h2,x2)
						xvals_h3 = np.append(xvals_h3,x3)
			
			l_fit1 = multi_linear([xvals_h1,xvals_h2,xvals_h3], *coeff_d[0])
			
			fig = plt.figure(figsize=(8, 5))
			ax = fig.add_subplot(111, projection='3d')
			
			x_lims = [lh[0],uh[0]]
			y_lims = [lh[1],uh[1]]
			z_lims = [lh[2],uh[2]]
			corner_coordinates = []
			for x in x_lims:
				for y in y_lims:
					for z in z_lims:
						corner_coordinates.append([x,y,z])
			
			faces = [
				[corner_coordinates[0], corner_coordinates[1], corner_coordinates[5], corner_coordinates[4]],  # Bottom face
				[corner_coordinates[2], corner_coordinates[3], corner_coordinates[7], corner_coordinates[6]],  # Top face
				[corner_coordinates[0], corner_coordinates[1], corner_coordinates[3], corner_coordinates[2]],  # Side face 1
				[corner_coordinates[1], corner_coordinates[5], corner_coordinates[7], corner_coordinates[3]],  # Side face 2
				[corner_coordinates[5], corner_coordinates[4], corner_coordinates[6], corner_coordinates[7]],  # Side face 3
				[corner_coordinates[4], corner_coordinates[0], corner_coordinates[2], corner_coordinates[6]],  # Side face 4
			]
			cube = Poly3DCollection(faces, facecolor='gray', linewidths=2, edgecolors='k', alpha=0.1, linestyles=':')
			ax.add_collection3d(cube)
			cmap_to_use = plt.cm.get_cmap('Blues')
			bounds = np.linspace((doses[0]-0.125)*100, (doses[-1]+0.125)*100, len(doses)+1)
			#ax.scatter(xvals_h1, xvals_h2, xvals_h3, c=l_fit1, cmap=cmap_to_use, s=1**2, vmin=doses[0], vmax=doses[-1], zorder=0, alpha=0.2)
			
			# thicker outlines of dots
			ax.scatter(data[0][sind][i][0], data[1][sind][i][0], data[2][sind][i][0], c='magenta', s=22**2, edgecolors='magenta') # Plot magenta color to outline zero dose
			for dind,_ in enumerate(data[0][sind][i]):
				if dind == 0: continue
				ax.scatter(data[0][sind][i][dind],data[1][sind][i][dind],data[2][sind][i][dind], c='k', s=22**2, edgecolors='k')
			
			img = ax.scatter(data[0][sind][i], data[1][sind][i], data[2][sind][i], c=[d*100 for d in doses], cmap=cmap_to_use, s=18**2, edgecolors='black', vmin=doses[0]*100, vmax=doses[-1]*100)
			ax.scatter(mh[0], mh[1], mh[2], c=MV_middle_dose_range*100, cmap=cmap_to_use, s=22**2, edgecolors='k', marker="d", linewidth=2, vmin=doses[0]*100, vmax=doses[-1]*100)
			#ax.scatter(uh[0], uh[1], uh[2], c=MV_lower_dose_range, cmap=cmap_to_use, s=12**2, edgecolors='k', marker="v", linewidth=2.5, vmin=doses[0], vmax=doses[-1])
			#ax.scatter(lh[0], lh[1], lh[2], c=MV_upper_dose_range, cmap=cmap_to_use, s=12**2, edgecolors='k', marker="^", linewidth=2.5, vmin=doses[0], vmax=doses[-1])
			#ax.add_collection3d(cube)
			#ax.view_init(-140, 60)
			ax.set_xlabel('\n\n1/f')
			ax.set_ylabel('\n\n'+r'$\theta$')
			ax.set_zlabel('\n'+r'$\alpha$')
			ax.set_xlim(np.nanmin(data[0][sind][i])-0.1,np.nanmax(data[0][sind][i])+0.1)
			ax.set_ylim(np.nanmin(data[1][sind][i])-0.1,np.nanmax(data[1][sind][i])+0.1)
			ax.set_zlim(np.nanmin(data[2][sind][i])-0.1,np.nanmax(data[2][sind][i])+0.1)
			ax.xaxis.set_major_locator(mticker.MultipleLocator(0.3)) # disable if using smaller axis values <1
			ax.yaxis.set_major_locator(mticker.MultipleLocator(0.3)) # disable if using smaller axis values <1
			ax.zaxis.set_major_locator(mticker.MultipleLocator(0.3)) # disable if using smaller axis values <1
			fig.colorbar(img,label='% of Reference Dose', ticks=[d*100 for d in doses], spacing='proportional', boundaries=bounds)
			fig.tight_layout()
			
			fig.savefig('figs_ManualLinearRegression_V13/opt_dose_fits/opt_dose_fit_severity'+str(s)+'_seed'+str(i)+'.png',dpi=300,transparent=True)
			plt.close()
			
			# Find optimal dose using individual metrics
			mdrs = []
			ldrs = []
			udrs = []
			for dataind,data in enumerate(all_data):
				data_doses = [data[dind][sind][i] for dind,d in enumerate(doses)]
				s1 = (doses[-1]-doses[0])/(data_doses[-1]-data_doses[0])
				i1 = np.mean([od - s1*d1 for od,d1 in zip(doses,data_doses)])
				p1 = [s1, i1]
				coeff_d = curve_fit(linear, data_doses, doses, p0 = p1, full_output=True)
				
				xvals_highres_d = np.linspace(np.min(doses),np.max(doses),10**3)
				l_fit1 = linear(xvals_highres_d, *coeff_d[0])
				
				# Note that the upper healthy SD corresponds to the lower dose range and vice versa when slope is negative
				if coeff_d[0][0] < 0:
					middle_dose_range = linear(np.array(mh[dataind]), *coeff_d[0])
					lower_dose_range = linear(np.array(uh[dataind]), *coeff_d[0])
					upper_dose_range = linear(np.array(lh[dataind]), *coeff_d[0])
				elif coeff_d[0][0] > 0:
					middle_dose_range = linear(np.array(mh[dataind]), *coeff_d[0])
					lower_dose_range = linear(np.array(lh[dataind]), *coeff_d[0])
					upper_dose_range = linear(np.array(uh[dataind]), *coeff_d[0])
				
				mdrs.append(middle_dose_range)
				ldrs.append(lower_dose_range)
				udrs.append(upper_dose_range)
			
			# Find upper and lower dose ranges based on average of intersection points for each metric
			mean_middle_dose_range = np.mean(mdrs)
			mean_lower_dose_range = np.mean(ldrs)
			mean_upper_dose_range = np.mean(udrs)
			# Find upper and lower dose ranges based on alpha metric only
			alpha_middle_dose_range = mdrs[2]
			alpha_lower_dose_range = ldrs[2]
			alpha_upper_dose_range = udrs[2]
			
			mmdr.append(MV_middle_dose_range)
			mldr.append(MV_lower_dose_range)
			mudr.append(MV_upper_dose_range)
	
	return mmdr, mldr, mudr

all_metrics = [AUC,
			AUC_t_abs,
			AUC_a_abs] #[metric][dose][severity][seed]
mean_healthy = [AUC_m[0][0],
			AUC_t_abs_m[0][0],
			AUC_a_abs_m[0][0]]
l_healthy = [AUC_lu[0][0][0],
			AUC_t_abs_lu[0][0][0],
			AUC_a_abs_lu[0][0][0]]
u_healthy = [AUC_lu[0][0][1],
			AUC_t_abs_lu[0][0][1],
			AUC_a_abs_lu[0][0][1]]

optimal_dose, optimal_dose_lower_limit, optimal_dose_upper_limit = find_optimal_dose_and_range(all_metrics,mean_healthy,l_healthy,u_healthy)

sd_healthy = [AUC_sd[0][0],AUC_sd[0][0],
			AUC_t_abs_sd[0][0],AUC_t_abs_sd[0][0],
			AUC_a_abs_sd[0][0],AUC_a_abs_sd[0][0]]

inputs = [list(itertools.chain.from_iterable(AUC[0])),
			list(itertools.chain.from_iterable(AUC_t_abs[0])),
			list(itertools.chain.from_iterable(AUC_a_abs[0])),
			list(itertools.chain.from_iterable(AUC_broad_abs[0]))]

MSE = []
MSE_linear0 = []
MSE_linear1 = []
MSE_linear2 = []
MSE_linear3 = []
test_percent_correct = []
test_percent_correct0 = []
test_percent_correct1 = []
test_percent_correct2 = []
test_percent_correct3 = []
test_residuals = []
test_residuals0 = []
test_residuals1 = []
test_residuals2 = []
test_residuals3 = []
test_pmDose_percent_correct = []
test_pmDose_percent_correct0 = []
test_pmDose_percent_correct1 = []
test_pmDose_percent_correct2 = []
test_pmDose_percent_correct3 = []
test_pmDose_percent_incorrect_above = []
test_pmDose_percent_incorrect0_above = []
test_pmDose_percent_incorrect1_above = []
test_pmDose_percent_incorrect2_above = []
test_pmDose_percent_incorrect3_above = []
test_pmDose_percent_incorrect_below = []
test_pmDose_percent_incorrect0_below = []
test_pmDose_percent_incorrect1_below = []
test_pmDose_percent_incorrect2_below = []
test_pmDose_percent_incorrect3_below = []
slopes_linear0_all = []
slopes_linear1_all = []
slopes_linear2_all = []
slopes_linear3_all = []
intercepts_linear0_all = []
intercepts_linear1_all = []
intercepts_linear2_all = []
intercepts_linear3_all = []

best_tpm = 0
best_tpm0 = 0
best_tpm1 = 0
best_tpm2 = 0
best_tpm3 = 0

all_mean_shaps = []

for p in range(0,NPermutes):
	print('Permutation #'+str(p))
	inputs_doses = sklearn.utils.shuffle(inputs[0],inputs[1],inputs[2],inputs[3],optimal_dose, optimal_dose_lower_limit, optimal_dose_upper_limit)
	inputs_doses_train = []
	inputs_doses_test = []
	for i in inputs_doses:
		inputs_doses_train.append(i[:int(len(i)*0.7)])
		inputs_doses_test.append(i[int(len(i)*0.7):])
	
	optimal_doses_middle = inputs_doses_test[-3]
	optimal_doses_lower = inputs_doses_test[-2]
	optimal_doses_upper = inputs_doses_test[-1]

	# train
	coeffs, fig_dosefits, coeffs_linear0, coeffs_linear1, coeffs_linear2, coeffs_linear3 = run_multi_line_fits(inputs_doses_train[:-3],inputs_doses_train[-3])
	
	# test
	test = multi_linear(np.array(inputs_doses_test[:-3]), *coeffs[0])
	test_linear0 = linear(np.array(inputs_doses_test[0]), *coeffs_linear0[0])
	test_linear1 = linear(np.array(inputs_doses_test[1]), *coeffs_linear1[0])
	test_linear2 = linear(np.array(inputs_doses_test[2]), *coeffs_linear2[0])
	test_linear3 = linear(np.array(inputs_doses_test[3]), *coeffs_linear3[0])
	
	def shap_multi_linear(x):
		return coeffs[0][0] + coeffs[0][1]*x[:,0] + coeffs[0][2]*x[:,1] + coeffs[0][3]*x[:,2]
	
	print('SHAP output:')
	vals = ['ap_AUC', 't_AUC', 'a_AUC']
	explainer = shap.KernelExplainer(shap_multi_linear, data = np.transpose(inputs_doses_train[:3]), feature_names=vals)
	shap_values = explainer(np.transpose(inputs_doses_test[:3]))
	mean_shaps = abs(shap_values.values).mean(axis=0)
	all_mean_shaps.append(mean_shaps)
	print(mean_shaps)
	
	MSE.append(np.mean((np.array(optimal_doses_middle)-np.array(test))**2))
	MSE_linear0.append(np.mean((np.array(optimal_doses_middle)-np.array(test_linear0))**2))
	MSE_linear1.append(np.mean((np.array(optimal_doses_middle)-np.array(test_linear1))**2))
	MSE_linear2.append(np.mean((np.array(optimal_doses_middle)-np.array(test_linear2))**2))
	MSE_linear3.append(np.mean((np.array(optimal_doses_middle)-np.array(test_linear3))**2))
	
	slopes_linear0_all.append(coeffs_linear0[0][0])
	slopes_linear1_all.append(coeffs_linear1[0][0])
	slopes_linear2_all.append(coeffs_linear2[0][0])
	slopes_linear3_all.append(coeffs_linear3[0][0])
	intercepts_linear0_all.append(coeffs_linear0[0][1])
	intercepts_linear1_all.append(coeffs_linear1[0][1])
	intercepts_linear2_all.append(coeffs_linear2[0][1])
	intercepts_linear3_all.append(coeffs_linear3[0][1])
	
	pred_dose = [min(doses, key=lambda x:abs(x-t)) for t in test]
	pred_dose0 = [min(doses, key=lambda x:abs(x-t)) for t in test_linear0]
	pred_dose1 = [min(doses, key=lambda x:abs(x-t)) for t in test_linear1]
	pred_dose2 = [min(doses, key=lambda x:abs(x-t)) for t in test_linear2]
	pred_dose3 = [min(doses, key=lambda x:abs(x-t)) for t in test_linear3]
	true_doses_rounded = [min(doses, key=lambda x:abs(x-idt)) for idt in optimal_doses_middle]
	
	# For this I now need to round both the predicted dose and the true dose to one of the simulated doses
	test_percent_correct.append((sum(np.array(pred_dose) == np.array(true_doses_rounded))/len(true_doses_rounded))*100)
	test_percent_correct0.append((sum(np.array(pred_dose0) == np.array(true_doses_rounded))/len(true_doses_rounded))*100)
	test_percent_correct1.append((sum(np.array(pred_dose1) == np.array(true_doses_rounded))/len(true_doses_rounded))*100)
	test_percent_correct2.append((sum(np.array(pred_dose2) == np.array(true_doses_rounded))/len(true_doses_rounded))*100)
	test_percent_correct3.append((sum(np.array(pred_dose3) == np.array(true_doses_rounded))/len(true_doses_rounded))*100)
	
	# Record residuals
	test_residuals.extend(np.abs(np.array(optimal_doses_middle)-np.array(test)))
	test_residuals0.extend(np.abs(np.array(optimal_doses_middle)-np.array(test_linear0)))
	test_residuals1.extend(np.abs(np.array(optimal_doses_middle)-np.array(test_linear1)))
	test_residuals2.extend(np.abs(np.array(optimal_doses_middle)-np.array(test_linear2)))
	test_residuals3.extend(np.abs(np.array(optimal_doses_middle)-np.array(test_linear3)))
	
	# Record percent correct, allowing for a upper/lower limit dose error range
	tpm = (np.sum([ol < t < ou for ol,t,ou in zip(optimal_doses_lower,test,optimal_doses_upper)]) / len(test))*100
	tpm0 = (np.sum([ol < t < ou for ol,t,ou in zip(optimal_doses_lower,test_linear0,optimal_doses_upper)]) / len(test_linear0))*100
	tpm1 = (np.sum([ol < t < ou for ol,t,ou in zip(optimal_doses_lower,test_linear1,optimal_doses_upper)]) / len(test_linear1))*100
	tpm2 = (np.sum([ol < t < ou for ol,t,ou in zip(optimal_doses_lower,test_linear2,optimal_doses_upper)]) / len(test_linear2))*100
	tpm3 = (np.sum([ol < t < ou for ol,t,ou in zip(optimal_doses_lower,test_linear3,optimal_doses_upper)]) / len(test_linear3))*100
	
	# Save best model across permutations
	if tpm > best_tpm:
		best_model = coeffs
		best_tpm = tpm
	if tpm0 > best_tpm0:
		best_model0 = coeffs_linear0
		best_tpm0 = tpm0
	if tpm1 > best_tpm1:
		best_model1 = coeffs_linear1
		best_tpm1 = tpm1
	if tpm2 > best_tpm2:
		best_model2 = coeffs_linear2
		best_tpm2 = tpm2
	if tpm3 > best_tpm3:
		best_model3 = coeffs_linear3
		best_tpm3 = tpm3
	
	test_pmDose_percent_correct.append(tpm)
	test_pmDose_percent_correct0.append(tpm0)
	test_pmDose_percent_correct1.append(tpm1)
	test_pmDose_percent_correct2.append(tpm2)
	test_pmDose_percent_correct3.append(tpm3)
	
	# Find % above and % below optimal dose ranges
	tpm_a = (np.sum([t >= ou for ol,t,ou in zip(optimal_doses_lower,test,optimal_doses_upper)]) / len(test))*100
	tpm0_a = (np.sum([t >= ou for ol,t,ou in zip(optimal_doses_lower,test_linear0,optimal_doses_upper)]) / len(test_linear0))*100
	tpm1_a = (np.sum([t >= ou for ol,t,ou in zip(optimal_doses_lower,test_linear1,optimal_doses_upper)]) / len(test_linear1))*100
	tpm2_a = (np.sum([t >= ou for ol,t,ou in zip(optimal_doses_lower,test_linear2,optimal_doses_upper)]) / len(test_linear2))*100
	tpm3_a = (np.sum([t >= ou for ol,t,ou in zip(optimal_doses_lower,test_linear3,optimal_doses_upper)]) / len(test_linear3))*100
	tpm_b = (np.sum([t <= ol for ol,t,ou in zip(optimal_doses_lower,test,optimal_doses_upper)]) / len(test))*100
	tpm0_b = (np.sum([t <= ol for ol,t,ou in zip(optimal_doses_lower,test_linear0,optimal_doses_upper)]) / len(test_linear0))*100
	tpm1_b = (np.sum([t <= ol for ol,t,ou in zip(optimal_doses_lower,test_linear1,optimal_doses_upper)]) / len(test_linear1))*100
	tpm2_b = (np.sum([t <= ol for ol,t,ou in zip(optimal_doses_lower,test_linear2,optimal_doses_upper)]) / len(test_linear2))*100
	tpm3_b = (np.sum([t <= ol for ol,t,ou in zip(optimal_doses_lower,test_linear3,optimal_doses_upper)]) / len(test_linear3))*100
	
	test_pmDose_percent_incorrect_above.append(tpm_a)
	test_pmDose_percent_incorrect0_above.append(tpm0_a)
	test_pmDose_percent_incorrect1_above.append(tpm1_a)
	test_pmDose_percent_incorrect2_above.append(tpm2_a)
	test_pmDose_percent_incorrect3_above.append(tpm3_a)
	test_pmDose_percent_incorrect_below.append(tpm_b)
	test_pmDose_percent_incorrect0_below.append(tpm0_b)
	test_pmDose_percent_incorrect1_below.append(tpm1_b)
	test_pmDose_percent_incorrect2_below.append(tpm2_b)
	test_pmDose_percent_incorrect3_below.append(tpm3_b)


fig_dosefits.tight_layout()
fig_dosefits.savefig('figs_ManualLinearRegression_V13/line_fits_severity_multi_linear_model.png',dpi=300,transparent=True)
plt.close()

cvec = []
for sind,s in enumerate(severities):
	for i in range(0,N_seeds):
		cvec.append(colors_conds_MDDonly[0][sind])

fig_dosefits, ax_dosefits = plt.subplots(1, 4, sharey=True, figsize=(20,5))

data = np.array(inputs)
#s1 = (np.max(optimal_dose)-np.min(optimal_dose))/(np.mean(data[0][(optimal_dose==np.max(optimal_dose))])-np.mean(data[0][(optimal_dose==np.min(optimal_dose))]))
#s2 = (np.max(optimal_dose)-np.min(optimal_dose))/(np.mean(data[1][(optimal_dose==np.max(optimal_dose))])-np.mean(data[1][(optimal_dose==np.min(optimal_dose))]))
#s3 = (np.max(optimal_dose)-np.min(optimal_dose))/(np.mean(data[2][(optimal_dose==np.max(optimal_dose))])-np.mean(data[2][(optimal_dose==np.min(optimal_dose))]))
#s4 = (np.max(optimal_dose)-np.min(optimal_dose))/(np.mean(data[3][(optimal_dose==np.max(optimal_dose))])-np.mean(data[3][(optimal_dose==np.min(optimal_dose))]))
#
#int0 = np.mean([od - s1*d1 - s2*d2 - s3*d3 - s4*d4 for od,d1,d2,d3,d4 in zip(optimal_dose,data[0],data[1],data[2],data[3])])
#int1 = np.mean([od - s1*d1 for od,d1 in zip(optimal_dose,data[0])])
#int2 = np.mean([od - s2*d2 for od,d2 in zip(optimal_dose,data[1])])
#int3 = np.mean([od - s3*d3 for od,d3 in zip(optimal_dose,data[2])])
#int4 = np.mean([od - s4*d4 for od,d4 in zip(optimal_dose,data[3])])
#
#p0_l = [int0, s1, s2, s3]
#p1 = [s1, int1]
#p2 = [s2, int2]
#p3 = [s3, int3]
#p4 = [s4, int4]
#
#coeff_l = curve_fit(multi_linear, data[:3], optimal_dose, p0 = p0_l, full_output=True)
#coeff_l_1 = curve_fit(linear, data[0], optimal_dose, p0 = p1, full_output=True)
#coeff_l_2 = curve_fit(linear, data[1], optimal_dose, p0 = p2, full_output=True)
#coeff_l_3 = curve_fit(linear, data[2], optimal_dose, p0 = p3, full_output=True)
#coeff_l_4 = curve_fit(linear, data[3], optimal_dose, p0 = p4, full_output=True)

# Use best models instead of overall fit (probably less overfitting risk this way)
coeff_l = best_model
coeff_l_1 = best_model0
coeff_l_2 = best_model1
coeff_l_3 = best_model2
coeff_l_4 = best_model3

def shap_multi_linear(x):
	return coeff_l[0][0] + coeff_l[0][1]*x[:,0] + coeff_l[0][2]*x[:,1] + coeff_l[0][3]*x[:,2]

print('SHAP output:')
vals = ['ap_AUC', 't_AUC', 'a_AUC']
explainer = shap.KernelExplainer(shap_multi_linear, data = np.transpose(inputs[:3]), feature_names=vals)
shap_values = explainer(np.transpose(inputs[:3]))
mean_shaps = abs(shap_values.values).mean(axis=0)
all_mean_shaps.append(mean_shaps)
print(mean_shaps)

median_shaps_linear = np.median(all_mean_shaps,axis=0)
mean_shaps_linear = np.mean(all_mean_shaps,axis=0)
sd_shaps_linear = np.std(all_mean_shaps,axis=0)
print(median_shaps_linear)
median_shaps_linear = [m for m in median_shaps_linear]
mean_shaps_linear = [m for m in mean_shaps_linear]
sd_shaps_linear = [s for s in sd_shaps_linear]
print(median_shaps_linear)

print('Median model SHAP outputs (across all ANNs):')
print(median_shaps_linear)
print('Mean model SHAP outputs (across all ANNs):')
print(mean_shaps_linear)
print('Stdev model SHAP outputs (across all ANNs):')
print(sd_shaps_linear)

xvals_shaps = [0,1,2]
fig_shaps, ax_shaps = plt.subplots(1, 1, figsize=(10,5))
ax_shaps.bar(xvals_shaps,mean_shaps_linear,yerr=sd_shaps_linear,width=0.7,color=['grey','grey','grey'],edgecolor='k',linewidth=3,error_kw=dict(lw=3, capsize=5, capthick=3))
ax_shaps.set_ylabel('Feature Importance\n(SHAP Value)')
ax_shaps.set_xticks(xvals_shaps)
ax_shaps.set_xticklabels(['AUC',r'$\theta$',r'$\alpha$'], rotation=45, ha='center')
ax_shaps.patch.set_facecolor('white')
fig_shaps.patch.set_alpha(0)
fig_shaps.tight_layout()
fig_shaps.savefig('figs_ManualLinearRegression_V13/shaps.png',dpi=300)
plt.close()


xvals_highres_1 = np.linspace(np.min(inputs[0]),np.max(inputs[0]),10**3) # use full dataset for plotting full range for each line
xvals_highres_2 = np.linspace(np.min(inputs[1]),np.max(inputs[1]),10**3)
xvals_highres_3 = np.linspace(np.min(inputs[2]),np.max(inputs[2]),10**3)
xvals_highres_4 = np.linspace(np.min(inputs[3]),np.max(inputs[3]),10**3)

l_fit1 = linear(xvals_highres_1, *coeff_l_1[0])
l_fit2 = linear(xvals_highres_2, *coeff_l_2[0])
l_fit3 = linear(xvals_highres_3, *coeff_l_3[0])
l_fit4 = linear(xvals_highres_4, *coeff_l_4[0])

od = optimal_dose
odl = optimal_dose_lower_limit
odu = optimal_dose_upper_limit

i0 = inputs[0]
i1 = inputs[1]
i2 = inputs[2]
i3 = inputs[3]

ax_dosefits[0].fill_between(sorted(i0), y1=[x for _,x in sorted(zip(i0,odl))], y2=[x for _,x in sorted(zip(i0,odu))], color='grey', alpha = 0.8)
ax_dosefits[1].fill_between(sorted(i1), y1=[x for _,x in sorted(zip(i1,odl))], y2=[x for _,x in sorted(zip(i1,odu))], color='grey', alpha = 0.8)
ax_dosefits[2].fill_between(sorted(i2), y1=[x for _,x in sorted(zip(i2,odl))], y2=[x for _,x in sorted(zip(i2,odu))], color='grey', alpha = 0.8)
ax_dosefits[3].fill_between(sorted(i3), y1=[x for _,x in sorted(zip(i3,odl))], y2=[x for _,x in sorted(zip(i3,odu))], color='grey', alpha = 0.8)
ax_dosefits[0].scatter(sorted(i0), [x for _,x in sorted(zip(i0,od))], c='k')
ax_dosefits[1].scatter(sorted(i1), [x for _,x in sorted(zip(i1,od))], c='k')
ax_dosefits[2].scatter(sorted(i2), [x for _,x in sorted(zip(i2,od))], c='k')
ax_dosefits[3].scatter(sorted(i3), [x for _,x in sorted(zip(i3,od))], c='k')
ax_dosefits[0].plot(xvals_highres_1, l_fit1, c='k')
ax_dosefits[1].plot(xvals_highres_2, l_fit2, c='k')
ax_dosefits[2].plot(xvals_highres_3, l_fit3, c='k')
ax_dosefits[3].plot(xvals_highres_4, l_fit4, c='k')

# Print Fits
print('Dose = '+f"{coeff_l_1[0][0]:0.2f}"+'AUC + '+f"{coeff_l_1[0][1]:0.2f}")
print('Dose = '+f"{coeff_l_2[0][0]:0.2f}"+'theta + '+f"{coeff_l_2[0][1]:0.2f}")
print('Dose = '+f"{coeff_l_3[0][0]:0.2f}"+'alpha + '+f"{coeff_l_3[0][1]:0.2f}")
print('Dose = '+f"{coeff_l_4[0][0]:0.2f}"+'(alpha+beta) + '+f"{coeff_l_4[0][1]:0.2f}")
print('Optimal Dose = '+f"{coeff_l[0][1]:0.2f}"+'AUC + '+f"{coeff_l[0][2]:0.2f}"+r'theta + '+f"{coeff_l[0][3]:0.2f}"+'alpha + '+f"{coeff_l[0][0]:0.2f}")

ax_dosefits[0].set_ylim(bottom=0)
ax_dosefits[0].set_xlabel('Aperiodic')
ax_dosefits[1].set_xlabel(r'$\theta$')
ax_dosefits[2].set_xlabel(r'$\alpha$')
ax_dosefits[3].set_xlabel(r'$\alpha$'+'+'+r'$\beta$')
ax_dosefits[0].set_ylabel('Optimal Dose Range')
fig_dosefits.tight_layout()
fig_dosefits.savefig('figs_ManualLinearRegression_V13/scatters_severity_optimaldose_withOverallFit.png',dpi=300,transparent=True)
plt.close()

# Find optimal dose using multivariate fits
xvals_highres_1 = np.linspace(np.nanmin(i0),np.nanmax(i0),5**2) # use full dataset for plotting full range for each line
xvals_highres_2 = np.linspace(np.nanmin(i1),np.nanmax(i1),5**2)
xvals_highres_3 = np.linspace(np.nanmin(i2),np.nanmax(i2),5**2)
xvals_h1 = np.array([])
xvals_h2 = np.array([])
xvals_h3 = np.array([])
for x1 in xvals_highres_1:
	for x2 in xvals_highres_2:
		for x3 in xvals_highres_3:
			xvals_h1 = np.append(xvals_h1,x1)
			xvals_h2 = np.append(xvals_h2,x2)
			xvals_h3 = np.append(xvals_h3,x3)

l_fit1 = multi_linear([xvals_h1,xvals_h2,xvals_h3], *coeff_l[0])
l_fit2 = multi_linear(np.array([i0,i1,i2]), *coeff_l[0])

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')

cmap_to_use = plt.cm.get_cmap('Blues')
#ax.scatter(xvals_h1, xvals_h2, xvals_h3, c=l_fit1, cmap=cmap_to_use, s=1**2, vmin=doses[0], vmax=doses[-1], zorder=0, alpha=0.2)

# Replaced "od" variable with "l_fit2" variable to plot the predicted optimal doses from the MV function instead of the estimated optimal doses.
img = ax.scatter(i0, i1, i2, c=[d*100 for d in l_fit2], cmap=cmap_to_use, s=15**2, edgecolors='black', linewidth = 2, vmin=doses[0]*100, vmax=doses[-1]*100, alpha=1)
ax.set_xlabel('\n\n1/f')
ax.set_ylabel('\n\n'+r'$\theta$')
ax.set_zlabel('\n'+r'$\alpha$')
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.3)) # disable if using smaller axis values <1
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.3)) # disable if using smaller axis values <1
ax.zaxis.set_major_locator(mticker.MultipleLocator(0.3)) # disable if using smaller axis values <1

fig.colorbar(img,label='% of Reference Dose',ticks=[d*100 for d in doses])
fig.tight_layout()

fig.savefig('figs_ManualLinearRegression_V13/scatters_severity_optimaldose_MVwithOverallFit.png',dpi=300,transparent=True)
plt.close()

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')

cmap_to_use = plt.cm.get_cmap('Blues')

# Plot overall fitted 4D function
#ax.scatter(xvals_h1, xvals_h2, xvals_h3, c=l_fit1, cmap=cmap_to_use, s=1**2, vmin=doses[0], vmax=doses[-1], zorder=0, alpha=0.2)

for i0_,i1_,i2_,odl_,odu_,od_ in zip(i0,i1,i2,odl,odu,od):
	img1 = ax.scatter(i0_, i1_, i2_, c=odl_*100, cmap=cmap_to_use, s=13**2, alpha=1, linewidths=0.5, edgecolor='k', marker=MarkerStyle("o", fillstyle="left"), vmin=doses[0]*100, vmax=doses[-1]*100)
	img2 = ax.scatter(i0_, i1_, i2_, c=odu_*100, cmap=cmap_to_use, s=13**2, alpha=1, linewidths=0.5, edgecolor='k', marker=MarkerStyle("o", fillstyle="right"), vmin=doses[0]*100, vmax=doses[-1]*100)
	img3 = ax.scatter(i0_, i1_, i2_, c=od_*100, cmap=cmap_to_use, s=7**2, alpha=1, linewidths=0.5, edgecolor='k', vmin=doses[0]*100, vmax=doses[-1]*100)
ax.set_xlabel('\n1/f')
ax.set_ylabel('\n'+r'$\theta$')
ax.set_zlabel(r'$\alpha$')
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.3)) # disable if using smaller axis values <1
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.3)) # disable if using smaller axis values <1
ax.zaxis.set_major_locator(mticker.MultipleLocator(0.3)) # disable if using smaller axis values <1

fig.colorbar(img1,label='Optimal Dose')
fig.tight_layout()

fig.savefig('figs_ManualLinearRegression_V13/scatters_severity_optimaldose_MVwithOverallFit_V2.png',dpi=300,transparent=True)
plt.close()


fig_dosefits, ax_dosefits = plt.subplots(1, 4, sharey=True, figsize=(20,5))
for i0, i1, i2, i3, od, odl, odu, col in zip(inputs[0],inputs[1],inputs[2],inputs[3],optimal_dose,optimal_dose_lower_limit,optimal_dose_upper_limit,cvec):
	ax_dosefits[0].scatter(i0, od, c=np.array([col]))
	ax_dosefits[1].scatter(i1, od, c=np.array([col]))
	ax_dosefits[2].scatter(i2, od, c=np.array([col]))
	ax_dosefits[3].scatter(i3, od, c=np.array([col]))
	ax_dosefits[0].errorbar(i0, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[1].errorbar(i1, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[2].errorbar(i2, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[3].errorbar(i3, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
ax_dosefits[0].set_xlabel('Aperiodic')
ax_dosefits[1].set_xlabel(r'$\theta$')
ax_dosefits[2].set_xlabel(r'$\alpha$')
ax_dosefits[3].set_xlabel(r'$\alpha$'+'+'+r'$\beta$')
ax_dosefits[0].set_ylabel('Optimal Dose')
fig_dosefits.tight_layout()
fig_dosefits.savefig('figs_ManualLinearRegression_V13/scatters_severity_optimaldose.png',dpi=300,transparent=True)
plt.close()

fig_dosefits, ax_dosefits = plt.subplots(1, 4, sharey=True, figsize=(20,5))
for i0, i1, i2, i3, od, odl, odu, col in zip(inputs[0][:20],inputs[1][:20],inputs[2][:20],inputs[3][:20],optimal_dose[:20],optimal_dose_lower_limit[:20],optimal_dose_upper_limit[:20],cvec[:20]):
	ax_dosefits[0].scatter(i0, od, c=np.array([col]))
	ax_dosefits[1].scatter(i1, od, c=np.array([col]))
	ax_dosefits[2].scatter(i2, od, c=np.array([col]))
	ax_dosefits[3].scatter(i3, od, c=np.array([col]))
	ax_dosefits[0].errorbar(i0, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[1].errorbar(i1, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[2].errorbar(i2, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[3].errorbar(i3, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
ax_dosefits[0].set_xlabel('Aperiodic')
ax_dosefits[1].set_xlabel(r'$\theta$')
ax_dosefits[2].set_xlabel(r'$\alpha$')
ax_dosefits[3].set_xlabel(r'$\alpha$'+'+'+r'$\beta$')
ax_dosefits[0].set_ylabel('Optimal Dose')
fig_dosefits.tight_layout()
fig_dosefits.savefig('figs_ManualLinearRegression_V13/scatters_severity_optimaldose_severity0.0.png',dpi=300,transparent=True)
plt.close()

fig_dosefits, ax_dosefits = plt.subplots(1, 4, sharey=True, figsize=(20,5))
for i0, i1, i2, i3, od, odl, odu, col in zip(inputs[0][20:40],inputs[1][20:40],inputs[2][20:40],inputs[3][20:40],optimal_dose[20:40],optimal_dose_lower_limit[20:40],optimal_dose_upper_limit[20:40],cvec[20:40]):
	ax_dosefits[0].scatter(i0, od, c=np.array([col]))
	ax_dosefits[1].scatter(i1, od, c=np.array([col]))
	ax_dosefits[2].scatter(i2, od, c=np.array([col]))
	ax_dosefits[3].scatter(i3, od, c=np.array([col]))
	ax_dosefits[0].errorbar(i0, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[1].errorbar(i1, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[2].errorbar(i2, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[3].errorbar(i3, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
ax_dosefits[0].set_xlabel('Aperiodic')
ax_dosefits[1].set_xlabel(r'$\theta$')
ax_dosefits[2].set_xlabel(r'$\alpha$')
ax_dosefits[3].set_xlabel(r'$\alpha$'+'+'+r'$\beta$')
ax_dosefits[0].set_ylabel('Optimal Dose')
fig_dosefits.tight_layout()
fig_dosefits.savefig('figs_ManualLinearRegression_V13/scatters_severity_optimaldose_severity0.1.png',dpi=300,transparent=True)
plt.close()

fig_dosefits, ax_dosefits = plt.subplots(1, 4, sharey=True, figsize=(20,5))
for i0, i1, i2, i3, od, odl, odu, col in zip(inputs[0][40:60],inputs[1][40:60],inputs[2][40:60],inputs[3][40:60],optimal_dose[40:60],optimal_dose_lower_limit[40:60],optimal_dose_upper_limit[40:60],cvec[40:60]):
	ax_dosefits[0].scatter(i0, od, c=np.array([col]))
	ax_dosefits[1].scatter(i1, od, c=np.array([col]))
	ax_dosefits[2].scatter(i2, od, c=np.array([col]))
	ax_dosefits[3].scatter(i3, od, c=np.array([col]))
	ax_dosefits[0].errorbar(i0, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[1].errorbar(i1, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[2].errorbar(i2, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[3].errorbar(i3, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
ax_dosefits[0].set_xlabel('Aperiodic')
ax_dosefits[1].set_xlabel(r'$\theta$')
ax_dosefits[2].set_xlabel(r'$\alpha$')
ax_dosefits[3].set_xlabel(r'$\alpha$'+'+'+r'$\beta$')
ax_dosefits[0].set_ylabel('Optimal Dose')
fig_dosefits.tight_layout()
fig_dosefits.savefig('figs_ManualLinearRegression_V13/scatters_severity_optimaldose_severity0.2.png',dpi=300,transparent=True)
plt.close()

fig_dosefits, ax_dosefits = plt.subplots(1, 4, sharey=True, figsize=(20,5))
for i0, i1, i2, i3, od, odl, odu, col in zip(inputs[0][60:80],inputs[1][60:80],inputs[2][60:80],inputs[3][60:80],optimal_dose[60:80],optimal_dose_lower_limit[60:80],optimal_dose_upper_limit[60:80],cvec[60:80]):
	ax_dosefits[0].scatter(i0, od, c=np.array([col]))
	ax_dosefits[1].scatter(i1, od, c=np.array([col]))
	ax_dosefits[2].scatter(i2, od, c=np.array([col]))
	ax_dosefits[3].scatter(i3, od, c=np.array([col]))
	ax_dosefits[0].errorbar(i0, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[1].errorbar(i1, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[2].errorbar(i2, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[3].errorbar(i3, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
ax_dosefits[0].set_xlabel('Aperiodic')
ax_dosefits[1].set_xlabel(r'$\theta$')
ax_dosefits[2].set_xlabel(r'$\alpha$')
ax_dosefits[3].set_xlabel(r'$\alpha$'+'+'+r'$\beta$')
ax_dosefits[0].set_ylabel('Optimal Dose')
fig_dosefits.tight_layout()
fig_dosefits.savefig('figs_ManualLinearRegression_V13/scatters_severity_optimaldose_severity0.3.png',dpi=300,transparent=True)
plt.close()

fig_dosefits, ax_dosefits = plt.subplots(1, 4, sharey=True, figsize=(20,5))
for i0, i1, i2, i3, od, odl, odu, col in zip(inputs[0][80:],inputs[1][80:],inputs[2][80:],inputs[3][80:],optimal_dose[80:],optimal_dose_lower_limit[80:],optimal_dose_upper_limit[80:],cvec[80:]):
	ax_dosefits[0].scatter(i0, od, c=np.array([col]))
	ax_dosefits[1].scatter(i1, od, c=np.array([col]))
	ax_dosefits[2].scatter(i2, od, c=np.array([col]))
	ax_dosefits[3].scatter(i3, od, c=np.array([col]))
	ax_dosefits[0].errorbar(i0, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[1].errorbar(i1, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[2].errorbar(i2, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
	ax_dosefits[3].errorbar(i3, od, yerr=[[od-odl],[odu-od]], ecolor=np.array([col]))
ax_dosefits[0].set_xlabel('Aperiodic')
ax_dosefits[1].set_xlabel(r'$\theta$')
ax_dosefits[2].set_xlabel(r'$\alpha$')
ax_dosefits[3].set_xlabel(r'$\alpha$'+'+'+r'$\beta$')
ax_dosefits[0].set_ylabel('Optimal Dose')
fig_dosefits.tight_layout()
fig_dosefits.savefig('figs_ManualLinearRegression_V13/scatters_severity_optimaldose_severity0.4.png',dpi=300,transparent=True)
plt.close()

min_overall = np.min([np.min(MSE),np.min(MSE_linear0),np.min(MSE_linear1),np.min(MSE_linear2),np.min(MSE_linear3)])
max_overall = np.max([np.max(MSE),np.max(MSE_linear0),np.max(MSE_linear1),np.max(MSE_linear2),np.max(MSE_linear3)])
bins_sequence = np.linspace(min_overall,max_overall,50)

fig_MSE, ax_MSES = plt.subplots(1,1, figsize=(16,5))
ax_MSES.hist(MSE,range=(min_overall,max_overall),bins=bins_sequence,color='grey',label='Dose=a+b*AUC'+'+c*'+r'$\theta$'+'+d*'+r'$\alpha$',alpha=0.4)
ax_MSES.hist(MSE_linear0,range=(min_overall,max_overall),bins=bins_sequence,color='tab:orange',label='Dose=a+b*AUC',alpha=0.4)
ax_MSES.hist(MSE_linear1,range=(min_overall,max_overall),bins=bins_sequence,color='tab:green',label='Dose=a+b*'+r'$\theta$',alpha=0.4)
ax_MSES.hist(MSE_linear2,range=(min_overall,max_overall),bins=bins_sequence,color='tab:blue',label='Dose=a+b*'+r'$\alpha$',alpha=0.4)
ax_MSES.hist(MSE_linear3,range=(min_overall,max_overall),bins=bins_sequence,color='tab:purple',label='Dose=a+b*('+r'$\alpha$'+'+'+r'$\beta$'+')',alpha=0.4)
ax_MSES.set_xlabel('Tested Mean Squared Error')
ax_MSES.set_ylabel('Count')
ax_MSES.legend(fontsize=20)
fig_MSE.tight_layout()
fig_MSE.savefig('figs_ManualLinearRegression_V13/MSE_multi_linear_model.png',dpi=300,transparent=True)
plt.close()

mean_percent_correct = [np.mean(test_percent_correct),np.mean(test_percent_correct0),np.mean(test_percent_correct1),np.mean(test_percent_correct2),np.mean(test_percent_correct3)]
sd_percent_correct = [np.std(test_percent_correct),np.std(test_percent_correct0),np.std(test_percent_correct1),np.std(test_percent_correct2),np.std(test_percent_correct3)]
xvals = np.linspace(1,len(mean_percent_correct),len(mean_percent_correct))

fig_percentcorrect, ax_percentcorrect = plt.subplots(1, 1, figsize=(15,5))
ax_percentcorrect.bar(xvals,mean_percent_correct,yerr=sd_percent_correct,width=0.7,color=['grey','grey','grey','grey','grey'],edgecolor='k',linewidth=3,error_kw=dict(lw=3, capsize=5, capthick=3))
ax_percentcorrect.set_ylabel('% Correct')
ax_percentcorrect.set_xticks(xvals)
ax_percentcorrect.set_xticklabels(['MV','1/f',r'$\theta$',r'$\alpha$',r'$\alpha$'+'+'+r'$\beta$'], rotation=45, ha='center')
ax_percentcorrect.set_yticks([0,25,50,75,100])
ax_percentcorrect.set_ylim(0,100)
ax_percentcorrect.patch.set_facecolor('white')
fig_percentcorrect.patch.set_alpha(0)
fig_percentcorrect.tight_layout()
fig_percentcorrect.savefig('figs_ManualLinearRegression_V13/percentcorrect.png',dpi=300)
plt.close()

mean_residuals = [np.mean(test_residuals),np.mean(test_residuals0),np.mean(test_residuals1),np.mean(test_residuals2)]
sd_residuals = [np.std(test_residuals),np.std(test_residuals0),np.std(test_residuals1),np.std(test_residuals2)]
xvals = np.linspace(1,len(mean_residuals),len(mean_residuals))

fig_residuals, ax_residuals = plt.subplots(1, 1, figsize=(6,5))
ax_residuals.bar(xvals,mean_residuals,yerr=sd_residuals,width=0.7,color=['grey','grey','grey','grey'],edgecolor='k',linewidth=3,error_kw=dict(lw=3, capsize=5, capthick=3))
ax_residuals.set_ylabel('Predicted Dose Error')
ax_residuals.set_xticks(xvals)
ax_residuals.set_xticklabels(['MV','1/f',r'$\theta$',r'$\alpha$'], rotation=45, ha='center')
ax_residuals.set_ylim(bottom=0)
ax_residuals.patch.set_facecolor('white')
fig_residuals.patch.set_alpha(0)
fig_residuals.tight_layout()
fig_residuals.savefig('figs_ManualLinearRegression_V13/residuals.png',dpi=300)
plt.close()

# Save accuracies to npy file for stats later
pmDose_all = [test_pmDose_percent_correct,test_pmDose_percent_correct0,test_pmDose_percent_correct1,test_pmDose_percent_correct2]
np.save('figs_ManualLinearRegression_V13/accuracy_pmDose_allvalues.npy',pmDose_all)
pmDose_all_above = [test_pmDose_percent_incorrect_above,test_pmDose_percent_incorrect0_above,test_pmDose_percent_incorrect1_above,test_pmDose_percent_incorrect2_above]
np.save('figs_ManualLinearRegression_V13/accuracy_pmDose_allvalues_above.npy',pmDose_all_above)
pmDose_all_below = [test_pmDose_percent_incorrect_below,test_pmDose_percent_incorrect0_below,test_pmDose_percent_incorrect1_below,test_pmDose_percent_incorrect2_below]
np.save('figs_ManualLinearRegression_V13/accuracy_pmDose_allvalues_below.npy',pmDose_all_below)

mean_pmDose_percent_correct = [np.mean(test_pmDose_percent_correct),np.mean(test_pmDose_percent_correct0),np.mean(test_pmDose_percent_correct1),np.mean(test_pmDose_percent_correct2)]
sd_pmDose_percent_correct = [np.std(test_pmDose_percent_correct),np.std(test_pmDose_percent_correct0),np.std(test_pmDose_percent_correct1),np.std(test_pmDose_percent_correct2)]
xvals = np.linspace(1,len(mean_pmDose_percent_correct),len(mean_pmDose_percent_correct))

fig_percentcorrect, ax_percentcorrect = plt.subplots(1, 1, figsize=(5,6))
ax_percentcorrect.bar(xvals,mean_pmDose_percent_correct,yerr=sd_pmDose_percent_correct,width=0.7,color=['grey','grey','grey','grey'],edgecolor='k',linewidth=3,error_kw=dict(lw=3, capsize=5, capthick=3))
ax_percentcorrect.set_ylabel('Dose Prediction Accuracy (%)')
ax_percentcorrect.set_xticks(xvals)
ax_percentcorrect.set_xticklabels(['MV','1/f',r'$\theta$',r'$\alpha$'], rotation=45, ha='center')
ax_percentcorrect.set_yticks([0,25,50,75,100])
ax_percentcorrect.set_ylim(0,100)
ax_percentcorrect.patch.set_facecolor('white')
fig_percentcorrect.patch.set_alpha(0)
fig_percentcorrect.tight_layout()
fig_percentcorrect.savefig('figs_ManualLinearRegression_V13/percentcorrect_pmDose.png',dpi=300)
plt.close()

mean_pmDose_percent_incorrect_above = [np.mean(test_pmDose_percent_incorrect_above),np.mean(test_pmDose_percent_incorrect0_above),np.mean(test_pmDose_percent_incorrect1_above),np.mean(test_pmDose_percent_incorrect2_above)]
sd_pmDose_percent_incorrect_above = [np.std(test_pmDose_percent_incorrect_above),np.std(test_pmDose_percent_incorrect0_above),np.std(test_pmDose_percent_incorrect1_above),np.std(test_pmDose_percent_incorrect2_above)]
xvals = np.linspace(1,len(mean_pmDose_percent_incorrect_above),len(mean_pmDose_percent_incorrect_above))

fig_percentincorrect, ax_percentincorrect = plt.subplots(1, 1, figsize=(5,6))
ax_percentincorrect.bar(xvals,mean_pmDose_percent_incorrect_above,yerr=sd_pmDose_percent_incorrect_above,width=0.7,color=['grey','grey','grey','grey'],edgecolor='k',linewidth=3,error_kw=dict(lw=3, capsize=5, capthick=3))
ax_percentincorrect.set_ylabel('% Above Range')
ax_percentincorrect.set_xticks(xvals)
ax_percentincorrect.set_xticklabels(['MV','1/f',r'$\theta$',r'$\alpha$'], rotation=45, ha='center')
ax_percentincorrect.set_yticks([0,5,10,15])
ax_percentincorrect.set_ylim(0,15)
ax_percentincorrect.patch.set_facecolor('white')
fig_percentincorrect.patch.set_alpha(0)
fig_percentincorrect.tight_layout()
fig_percentincorrect.savefig('figs_ManualLinearRegression_V13/percentincorrect_pmDose_above.png',dpi=300)
plt.close()

mean_pmDose_percent_incorrect_below = [np.mean(test_pmDose_percent_incorrect_below),np.mean(test_pmDose_percent_incorrect0_below),np.mean(test_pmDose_percent_incorrect1_below),np.mean(test_pmDose_percent_incorrect2_below)]
sd_pmDose_percent_incorrect_below = [np.std(test_pmDose_percent_incorrect_below),np.std(test_pmDose_percent_incorrect0_below),np.std(test_pmDose_percent_incorrect1_below),np.std(test_pmDose_percent_incorrect2_below)]
xvals = np.linspace(1,len(mean_pmDose_percent_incorrect_below),len(mean_pmDose_percent_incorrect_below))

fig_percentincorrect, ax_percentincorrect = plt.subplots(1, 1, figsize=(5,6))
ax_percentincorrect.bar(xvals,mean_pmDose_percent_incorrect_below,yerr=sd_pmDose_percent_incorrect_below,width=0.7,color=['grey','grey','grey','grey'],edgecolor='k',linewidth=3,error_kw=dict(lw=3, capsize=5, capthick=3))
ax_percentincorrect.set_ylabel('% Below Range')
ax_percentincorrect.set_xticks(xvals)
ax_percentincorrect.set_xticklabels(['MV','1/f',r'$\theta$',r'$\alpha$'], rotation=45, ha='center')
ax_percentincorrect.set_yticks([0,5,10,15])
ax_percentincorrect.set_ylim(0,15)
ax_percentincorrect.patch.set_facecolor('white')
fig_percentincorrect.patch.set_alpha(0)
fig_percentincorrect.tight_layout()
fig_percentincorrect.savefig('figs_ManualLinearRegression_V13/percentincorrect_pmDose_below.png',dpi=300)
plt.close()

fig_percentcorrect, ax_percentcorrect = plt.subplots(1, 1, figsize=(5,6))
ax_percentcorrect.bar(xvals,mean_pmDose_percent_correct,yerr=sd_pmDose_percent_correct,width=0.7,color=['grey','grey','grey','grey'],edgecolor='k',linewidth=3,error_kw=dict(lw=3, capsize=5, capthick=3))
ax_percentcorrect.bar(xvals-0.35/2,mean_pmDose_percent_incorrect_below,yerr=sd_pmDose_percent_incorrect_below,width=0.35,color=['tab:blue','tab:blue','tab:blue','tab:blue'],edgecolor='k',linewidth=3,error_kw=dict(lw=3, capsize=5, capthick=3))
ax_percentcorrect.bar(xvals+0.35/2,mean_pmDose_percent_incorrect_above,yerr=sd_pmDose_percent_incorrect_above,width=0.35,color=['tab:red','tab:red','tab:red','tab:red'],edgecolor='k',linewidth=3,error_kw=dict(lw=3, capsize=5, capthick=3))
ax_percentcorrect.set_ylabel('Dose Prediction Accuracy (%)')
ax_percentcorrect.set_xticks(xvals)
ax_percentcorrect.set_xticklabels(['MV','1/f',r'$\theta$',r'$\alpha$'], rotation=45, ha='center')
ax_percentcorrect.set_yticks([0,25,50,75,100])
ax_percentcorrect.set_ylim(0,100)
ax_percentcorrect.patch.set_facecolor('white')
fig_percentcorrect.patch.set_alpha(0)
fig_percentcorrect.tight_layout()
fig_percentcorrect.savefig('figs_ManualLinearRegression_V13/percentcorrect_pmDose_withabovebelow.png',dpi=300)
plt.close()
print('Accuracy Score:')
print(mean_pmDose_percent_correct)

median_slopes_linear = [np.median(slopes_linear0_all),np.median(slopes_linear1_all),np.median(slopes_linear2_all),np.median(slopes_linear3_all)]
mean_slopes_linear = [np.mean(slopes_linear0_all),np.mean(slopes_linear1_all),np.mean(slopes_linear2_all),np.mean(slopes_linear3_all)]
sd_slopes_linear = [np.std(slopes_linear0_all),np.std(slopes_linear1_all),np.std(slopes_linear2_all),np.std(slopes_linear3_all)]
xvals_slopes = np.linspace(1,len(mean_slopes_linear),len(mean_slopes_linear))

print(['AUC',r'$\theta$',r'$\alpha$',r'$\alpha$'+'+'+r'$\beta$'])
print('Median model beta coefficients (across all models)')
print(median_slopes_linear)
print('Mean model beta coefficients (across all models)')
print(mean_slopes_linear)
print('Stdev model beta coefficients (across all models)')
print(sd_slopes_linear)

mean_intercepts_linear = [np.mean(intercepts_linear0_all),np.mean(intercepts_linear1_all),np.mean(intercepts_linear2_all),np.mean(intercepts_linear3_all)]
sd_intercepts_linear = [np.std(intercepts_linear0_all),np.std(intercepts_linear1_all),np.std(intercepts_linear2_all),np.std(intercepts_linear3_all)]
xvals_intercepts = np.linspace(1,len(mean_intercepts_linear),len(mean_intercepts_linear))

fig_slopes, ax_slopes = plt.subplots(1, 1, figsize=(13,5))
ax_slopes.bar(xvals_slopes,mean_slopes_linear,yerr=sd_slopes_linear,width=0.7,color=['grey','grey','grey','grey'],edgecolor='k',linewidth=3,error_kw=dict(lw=3, capsize=5, capthick=3))
ax_slopes.set_ylabel('Slope (Dose/PSD'+r'$^{2}$'+')')
ax_slopes.set_xticks(xvals_slopes)
ax_slopes.set_xticklabels(['AUC',r'$\theta$',r'$\alpha$',r'$\alpha$'+'+'+r'$\beta$'], rotation=45, ha='center')
ax_slopes.patch.set_facecolor('white')
fig_slopes.patch.set_alpha(0)
fig_slopes.tight_layout()
fig_slopes.savefig('figs_ManualLinearRegression_V13/slopes.png',dpi=300)
plt.close()

fig_intercepts, ax_intercepts = plt.subplots(1, 1, figsize=(13,5))
ax_intercepts.bar(xvals_intercepts,mean_intercepts_linear,yerr=sd_intercepts_linear,width=0.7,color=['grey','grey','grey','grey'],edgecolor='k',linewidth=3,error_kw=dict(lw=3, capsize=5, capthick=3))
ax_intercepts.set_ylabel('Intercept (Dose)')
ax_intercepts.set_xticks(xvals_intercepts)
ax_intercepts.set_xticklabels(['AUC',r'$\theta$',r'$\alpha$',r'$\alpha$'+'+'+r'$\beta$'], rotation=45, ha='center')
ax_intercepts.patch.set_facecolor('white')
fig_intercepts.patch.set_alpha(0)
fig_intercepts.tight_layout()
fig_intercepts.savefig('figs_ManualLinearRegression_V13/intercepts.png',dpi=300)
plt.close()