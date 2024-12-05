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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import (LogLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.lines import Line2D
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
import math as math
import scipy.special as sp
import LFPy
import pandas as pd
from fooof import FOOOF
from copy import copy
import itertools
import sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
import tensorflow
import keras.backend as K

np.random.seed(seed=1234)

def skewnorm(x, sigmag, mu, alpha,a):
	#normal distribution
	normpdf = (1/(sigmag*np.sqrt(2*math.pi)))*np.exp(-(np.power((x-mu),2)/(2*np.power(sigmag,2))))
	normcdf = (0.5*(1+sp.erf((alpha*((x-mu)/sigmag))/(np.sqrt(2)))))
	return 2*a*normpdf*normcdf

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':26})

NPermutes = 10
N_seeds = 20
N_seedsList = np.linspace(1,N_seeds,N_seeds, dtype=int)
N_cells = 1000
N_HL23PN = int(0.8*N_cells)
N_HL23MN = int(0.05*N_cells)
N_HL23BN = int(0.07*N_cells)
N_HL23VN = int(0.08*N_cells)
rate_threshold = 0.2 # Hz
dt = 0.025 # ms
startslice = 1000 # ms
endslice = 25000 # ms
response_window = 50
t1 = int(startslice*(1/dt))
t2 = int(endslice*(1/dt))
tvec = np.arange(endslice/dt+1)*dt
nperseg = 80000 # len(tvec[t1:t2])/2
rate_window = 5.000001 # Hz
ratebins = np.arange(0,rate_window,0.2)
bin_centres_highres = np.arange(0.001,rate_window,0.001)

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
paths_rates = [['Saved_SpikesOnly/'+i+'/' for i in c] for c in conds]

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

# Extract failed/false detection and spike rate metrics
base_rates = [[[] for s in severities] for d in doses]
failed_detections = [[[] for s in severities] for d in doses]
false_detections = [[[] for s in severities] for d in doses]

post_fit = np.load('figs_ManualLinearRegression_PSDvalidation_V5_ANN/post_stimulus_default_circuit.npy')

for seed in N_seedsList:
	for cind1, path1 in enumerate(paths_rates):
		for cind2, path2 in enumerate(path1):
			print('Analyzing seed #'+str(seed)+' for '+conds[cind1][cind2])
			
			try:
				temp_s = np.load(paths_rates[cind1][cind2] + 'SPIKES_Seed' + str(seed) + '.npy',allow_pickle=True)
			except:
				base_rates[cind1][cind2].append(np.nan)
				failed_detections[cind1][cind2].append(np.nan)
				false_detections[cind1][cind2].append(np.nan)
				print(paths_rates[cind1][cind2] + ' does not exist')
				continue
			
			SPIKES = temp_s.item()
			temp_rn = []
			for i in range(0,len(SPIKES['times'][0])):
				scount = SPIKES['times'][0][i][(SPIKES['times'][0][i]>startslice) & (SPIKES['times'][0][i]<=endslice)]
				Hz = (scount.size)/((int(endslice)-startslice)/1000)
				temp_rn.append(Hz)
			
			base_rates[cind1][cind2].append(np.mean(temp_rn))
			
			SPIKES1 = [x for _,x in sorted(zip(temp_s.item()['gids'][0],temp_s.item()['times'][0]))]
			popspikes = np.concatenate(SPIKES1).ravel()
			spikeratevec = []
			for slide in np.arange(0,response_window,1):
				pre_bins = np.arange(startslice+slide,endslice+dt,response_window)
				spikeratevec.extend((np.histogram(popspikes,bins=pre_bins)[0]/(response_window/1000))/N_HL23PN)
			
			# Fit baseline spike rate distribution to Gaussian
			s_hist, s_bins = np.histogram(spikeratevec,bins = ratebins, density=True)
			bin_centres = (s_bins[:-1] + s_bins[1:])/2
			p0 = [np.std(s_hist), np.mean(s_hist), 1, 1]
			coeff, var_matrix = curve_fit(skewnorm, bin_centres, s_hist, p0 = p0, bounds = [0,(np.std(s_hist)*5,np.mean(s_hist)*10,2,200)], maxfev = 100000000)
			pre_fit = skewnorm(bin_centres_highres, *coeff)
			
			idx = np.argwhere(np.diff(np.sign(pre_fit - post_fit))).flatten()
			idx = idx[np.where(np.logical_and(idx>200, idx<=2200))][0]
			failed_detections[cind1][cind2].append((np.trapz(post_fit[:idx])/np.trapz(post_fit))*100)
			false_detections[cind1][cind2].append((np.trapz(pre_fit[idx:])/np.trapz(pre_fit))*100)

base_rates_m = [[np.mean(allvals) for allvals in a] for a in base_rates]
base_rates_sd = [[np.std(allvals) for allvals in a] for a in base_rates]
failed_detections_m = [[np.mean(allvals) for allvals in a] for a in failed_detections]
failed_detections_sd = [[np.std(allvals) for allvals in a] for a in failed_detections]
false_detections_m = [[np.mean(allvals) for allvals in a] for a in false_detections]
false_detections_sd = [[np.std(allvals) for allvals in a] for a in false_detections]

percentiles0 = [0,100] # upper and lower percentiles to compute
base_rates_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in base_rates]
failed_detections_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in failed_detections]
false_detections_lu = [[np.percentile(allvals,percentiles0) for allvals in a] for a in false_detections]

# vs healthy
base_rates_tstat0 = [[st.ttest_rel(base_rates[0][0],base_rates[d][s])[0] if base_rates[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
base_rates_p0 = [[st.ttest_rel(base_rates[0][0],base_rates[d][s])[1] if base_rates[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
base_rates_cd0 = [[cohen_d(base_rates[0][0],base_rates[d][s]) if base_rates[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
failed_detections_tstat0 = [[st.ttest_rel(failed_detections[0][0],failed_detections[d][s])[0] if failed_detections[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
failed_detections_p0 = [[st.ttest_rel(failed_detections[0][0],failed_detections[d][s])[1] if failed_detections[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
failed_detections_cd0 = [[cohen_d(failed_detections[0][0],failed_detections[d][s]) if failed_detections[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
false_detections_tstat0 = [[st.ttest_rel(false_detections[0][0],false_detections[d][s])[0] if false_detections[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
false_detections_p0 = [[st.ttest_rel(false_detections[0][0],false_detections[d][s])[1] if false_detections[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]
false_detections_cd0 = [[cohen_d(false_detections[0][0],false_detections[d][s]) if false_detections[d][s][0] is not np.nan else np.nan for s in range(0,len(severities))] for d in range(0,len(doses))]

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

# Fit functions to dose-curves
def linear(x, slope, intercept):
	return slope*x + intercept

def expdec(x, a, k, b):
    return a * np.exp(-k*x) + b

def multi_linear(x, a, b, c, d):
    return a + b*x[0] + c*x[1] + d*x[2]

def linear_reverse(y, slope, intercept):
	return (y - intercept)/slope

def expdec_reverse(y, a, k, b):
    return -np.log((y - b)/a)/k

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
			#mmdr.append(MV_middle_dose_range if MV_middle_dose_range>=0 else 0)
			#mldr.append(MV_lower_dose_range if MV_lower_dose_range>=0 else 0)
			#mudr.append(MV_upper_dose_range if MV_upper_dose_range>=0 else 0)

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

# Find optimal doses (based on rate for each individual
def find_optimal_dose_and_range_rate(all_data,mh,lh,uh):
	mmdr = []
	mldr = []
	mudr = []
	for sind,s in enumerate(severities):
		for i in range(0,N_seeds):
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
					upper_dose_range = linear(np.array(hh[dataind]), *coeff_d[0])
				
				mdrs.append(middle_dose_range)
				ldrs.append(lower_dose_range)
				udrs.append(upper_dose_range)
			
			# Find upper and lower dose ranges based on average of intersection points for each metric
			mean_middle_dose_range = np.mean(mdrs)
			mean_lower_dose_range = np.mean(ldrs)
			mean_upper_dose_range = np.mean(udrs)
			
			mmdr.append(mean_middle_dose_range)
			mldr.append(mean_lower_dose_range)
			mudr.append(mean_upper_dose_range)
			#mmdr.append(mean_middle_dose_range if mean_middle_dose_range>=0 else 0)
			#mldr.append(mean_lower_dose_range if mean_lower_dose_range>=0 else 0)
			#mudr.append(mean_upper_dose_range if mean_upper_dose_range>=0 else 0)

	return mmdr, mldr, mudr

all_metrics_rate = [base_rates] #[metric][dose][severity][seed]
mean_healthy_rate = [base_rates_m[0][0]]
l_healthy_rate = [base_rates_lu[0][0][0]]
u_healthy_rate = [base_rates_lu[0][0][1]]

optimal_dose_rate, optimal_dose_lower_limit_rate, optimal_dose_upper_limit_rate = find_optimal_dose_and_range_rate(all_metrics_rate,mean_healthy_rate,l_healthy_rate,u_healthy_rate)

sd_healthy_rate = [base_rates_sd[0][0],base_rates_sd[0][0]]

# Create inputs list
inputs = [list(itertools.chain.from_iterable(AUC[0])),
			list(itertools.chain.from_iterable(AUC_t_abs[0])),
			list(itertools.chain.from_iterable(AUC_a_abs[0])),
			list(itertools.chain.from_iterable(base_rates[0]))]

cvec = []
for sind,s in enumerate(severities):
	for i in range(0,N_seeds):
		cvec.append(colors_conds_MDDonly[0][sind])

sev_list = [[x]*N_seeds for x in severities]
sev_list = list(itertools.chain.from_iterable(sev_list))
seedslist = list(N_seedsList)*len(severities)

best_tpm0 = 0
best_tpm1 = 0
best_tpm2 = 0
best_tpm3 = 0
best_tpm4 = 0

for p in range(0,NPermutes):
	print('Permutation #'+str(p))
	inputs_doses = sklearn.utils.shuffle(inputs[0],inputs[1],inputs[2],inputs[3],sev_list,seedslist,optimal_dose_rate, optimal_dose_lower_limit_rate, optimal_dose_upper_limit_rate,optimal_dose, optimal_dose_lower_limit, optimal_dose_upper_limit)
	inputs_doses_train = []
	inputs_doses_test = []
	for i in inputs_doses:
		inputs_doses_train.append(i[:int(len(i)*0.7)])
		inputs_doses_test.append(i[int(len(i)*0.7):])
	
	optimal_doses_middle_r = inputs_doses_test[-6]
	optimal_doses_lower_r = inputs_doses_test[-5]
	optimal_doses_upper_r = inputs_doses_test[-4]
	optimal_doses_middle = inputs_doses_test[-3]
	optimal_doses_lower = inputs_doses_test[-2]
	optimal_doses_upper = inputs_doses_test[-1]
	
	checkpoint_filepath = 'figs_ManualLinearRegression_PSDvalidation_V5_ANN/checkpoint.model.keras'
	checkpoint_filepath1 = 'figs_ManualLinearRegression_PSDvalidation_V5_ANN/checkpoint1.model.keras'
	checkpoint_filepath2 = 'figs_ManualLinearRegression_PSDvalidation_V5_ANN/checkpoint2.model.keras'
	checkpoint_filepath3 = 'figs_ManualLinearRegression_PSDvalidation_V5_ANN/checkpoint3.model.keras'
	checkpoint_filepath4 = 'figs_ManualLinearRegression_PSDvalidation_V5_ANN/checkpoint4.model.keras'
	model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
													filepath=checkpoint_filepath,
													save_weights_only=True,
													monitor='val_mae',
													mode='min',
													save_best_only=True)
	model_checkpoint_callback1 = keras.callbacks.ModelCheckpoint(
													filepath=checkpoint_filepath1,
													save_weights_only=True,
													monitor='val_mae',
													mode='min',
													save_best_only=True)
	model_checkpoint_callback2 = keras.callbacks.ModelCheckpoint(
													filepath=checkpoint_filepath2,
													save_weights_only=True,
													monitor='val_mae',
													mode='min',
													save_best_only=True)
	model_checkpoint_callback3 = keras.callbacks.ModelCheckpoint(
													filepath=checkpoint_filepath3,
													save_weights_only=True,
													monitor='val_mae',
													mode='min',
													save_best_only=True)
	model_checkpoint_callback4 = keras.callbacks.ModelCheckpoint(
													filepath=checkpoint_filepath4,
													save_weights_only=True,
													monitor='val_mae',
													mode='min',
													save_best_only=True)
	
	# train
	initializer = tensorflow.keras.initializers.HeNormal()
	model = Sequential()
	model1 = Sequential()
	model2 = Sequential()
	model3 = Sequential()
	model4 = Sequential()
	model.add(Dense(9, input_dim=3, activation='relu', kernel_initializer=initializer))
	model1.add(Dense(9, input_dim=1, activation='relu', kernel_initializer=initializer))
	model2.add(Dense(9, input_dim=1, activation='relu', kernel_initializer=initializer))
	model3.add(Dense(9, input_dim=1, activation='relu', kernel_initializer=initializer))
	model4.add(Dense(9, input_dim=1, activation='relu', kernel_initializer=initializer))
	model.add(Dense(1, activation='linear', kernel_initializer=initializer))
	model1.add(Dense(1, activation='linear', kernel_initializer=initializer))
	model2.add(Dense(1, activation='linear', kernel_initializer=initializer))
	model3.add(Dense(1, activation='linear', kernel_initializer=initializer))
	model4.add(Dense(1, activation='linear', kernel_initializer=initializer))
	#opt = keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9)
	opt = keras.optimizers.legacy.Adam(learning_rate=0.01)
	model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mae'])
	model1.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mae'])
	model2.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mae'])
	model3.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mae'])
	model4.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mae'])
	
	model.fit(list(zip(*inputs_doses_train[:3])),inputs_doses_train[-3], epochs=1000, batch_size=12, callbacks=[model_checkpoint_callback],validation_data=(list(zip(*inputs_doses_test[:3])), inputs_doses_test[-3]))
	model1.fit(inputs_doses_train[0],inputs_doses_train[-3], epochs=1000, batch_size=12, callbacks=[model_checkpoint_callback1],validation_data=(inputs_doses_test[0], inputs_doses_test[-3]))
	model2.fit(inputs_doses_train[1],inputs_doses_train[-3], epochs=1000, batch_size=12, callbacks=[model_checkpoint_callback2],validation_data=(inputs_doses_test[1], inputs_doses_test[-3]))
	model3.fit(inputs_doses_train[2],inputs_doses_train[-3], epochs=1000, batch_size=12, callbacks=[model_checkpoint_callback3],validation_data=(inputs_doses_test[2], inputs_doses_test[-3]))
	model4.fit(inputs_doses_train[3],inputs_doses_train[-6], epochs=1000, batch_size=12, callbacks=[model_checkpoint_callback4],validation_data=(inputs_doses_test[3], inputs_doses_test[-6]))
	
	# Compare best models for each permutation
	model.load_weights(checkpoint_filepath)
	model1.load_weights(checkpoint_filepath1)
	model2.load_weights(checkpoint_filepath2)
	model3.load_weights(checkpoint_filepath3)
	model4.load_weights(checkpoint_filepath4)
	
	predictions = model.predict(list(zip(*inputs_doses_test[:3])))
	predictions1 = model1.predict(inputs_doses_test[0])
	predictions2 = model2.predict(inputs_doses_test[1])
	predictions3 = model3.predict(inputs_doses_test[2])
	predictions4 = model4.predict(inputs_doses_test[3])
	
	# Record percent correct, allowing for a upper/lower limit dose error range
	tpm0 = (np.sum([ol < t < ou for ol,t,ou in zip(optimal_doses_lower,predictions,optimal_doses_upper)]) / len(predictions))*100
	tpm1 = (np.sum([ol < t < ou for ol,t,ou in zip(optimal_doses_lower,predictions1,optimal_doses_upper)]) / len(predictions1))*100
	tpm2 = (np.sum([ol < t < ou for ol,t,ou in zip(optimal_doses_lower,predictions2,optimal_doses_upper)]) / len(predictions2))*100
	tpm3 = (np.sum([ol < t < ou for ol,t,ou in zip(optimal_doses_lower,predictions3,optimal_doses_upper)]) / len(predictions3))*100
	tpm4 = (np.sum([ol < t < ou for ol,t,ou in zip(optimal_doses_lower_r,predictions4,optimal_doses_upper_r)]) / len(predictions4))*100
	
	# Save best model across permutations
	if tpm0 > best_tpm0:
		best_model0 = model
		best_train0 = inputs_doses_train
		best_test0 = inputs_doses_test
		best_tpm0 = tpm0
	if tpm1 > best_tpm1:
		best_model1 = model1
		best_train1 = inputs_doses_train
		best_test1 = inputs_doses_test
		best_tpm1 = tpm1
	if tpm2 > best_tpm2:
		best_model2 = model2
		best_train2 = inputs_doses_train
		best_test2 = inputs_doses_test
		best_tpm2 = tpm2
	if tpm3 > best_tpm3:
		best_model3 = model3
		best_train3 = inputs_doses_train
		best_test3 = inputs_doses_test
		best_tpm3 = tpm3
	if tpm4 > best_tpm4:
		best_model4 = model4
		best_train4 = inputs_doses_train
		best_test4 = inputs_doses_test
		best_tpm4 = tpm4

y_lin = best_model0.predict(list(zip(*best_test0[:3])))
y_lin0 = best_model1.predict(best_test1[0])
y_lin1 = best_model2.predict(best_test2[1])
y_lin2 = best_model3.predict(best_test3[2])
y_lin3 = best_model4.predict(best_test4[3])

# Replace negative predictions with 0
test = [t[0] if t>=0 else 0 for t in y_lin]
test_linear0 = [t[0] if t>=0 else 0 for t in y_lin0]
test_linear1 = [t[0] if t>=0 else 0 for t in y_lin1]
test_linear2 = [t[0] if t>=0 else 0 for t in y_lin2]
test_linear3 = [t[0] if t>=0 else 0 for t in y_lin3]

od = optimal_dose
odl = optimal_dose_lower_limit
odu = optimal_dose_upper_limit

odr = optimal_dose_rate
odrl = optimal_dose_lower_limit_rate
odru = optimal_dose_upper_limit_rate

i0 = inputs[0]
i1 = inputs[1]
i2 = inputs[2]
i3 = inputs[3]

fig_dosefits, ax_dosefits = plt.subplots(1, 4, sharey=True, figsize=(20,5))
ax_dosefits[0].fill_between(sorted(i0), y1=[x for _,x in sorted(zip(i0,odl))], y2=[x for _,x in sorted(zip(i0,odu))], color='grey', alpha = 0.8)
ax_dosefits[1].fill_between(sorted(i1), y1=[x for _,x in sorted(zip(i1,odl))], y2=[x for _,x in sorted(zip(i1,odu))], color='grey', alpha = 0.8)
ax_dosefits[2].fill_between(sorted(i2), y1=[x for _,x in sorted(zip(i2,odl))], y2=[x for _,x in sorted(zip(i2,odu))], color='grey', alpha = 0.8)
ax_dosefits[3].fill_between(sorted(i3), y1=[x for _,x in sorted(zip(i3,odrl))], y2=[x for _,x in sorted(zip(i3,odru))], color='grey', alpha = 0.8)
ax_dosefits[0].scatter(sorted(i0), [x for _,x in sorted(zip(i0,od))], c='k')
ax_dosefits[1].scatter(sorted(i1), [x for _,x in sorted(zip(i1,od))], c='k')
ax_dosefits[2].scatter(sorted(i2), [x for _,x in sorted(zip(i2,od))], c='k')
ax_dosefits[3].scatter(sorted(i3), [x for _,x in sorted(zip(i3,odr))], c='k')
ax_dosefits[0].set_ylim(bottom=0)
ax_dosefits[0].set_xlabel('Aperiodic')
ax_dosefits[1].set_xlabel(r'$\theta$')
ax_dosefits[2].set_xlabel(r'$\alpha$')
ax_dosefits[3].set_xlabel('Spike Rate (Hz)')
ax_dosefits[0].set_ylabel('Optimal Dose Range')
fig_dosefits.tight_layout()
fig_dosefits.savefig('figs_ManualLinearRegression_PSDvalidation_V5_ANN/scatters_severity_optimaldose_withOverallFit.png',dpi=300,transparent=True)
plt.close()

def get_dose_response_fits(cind2_,seedind_):
	data_doses_rate =[x[cind2_][seedind_] for x in base_rates]
	data_doses_failed =[x[cind2_][seedind_] for x in failed_detections]
	data_doses_false =[x[cind2_][seedind_] for x in false_detections]
	s1_rate = (doses[-1]-doses[0])/(data_doses_rate[-1]-data_doses_rate[0])
	i1_rate = np.mean([od - s1_rate*d1 for od,d1 in zip(doses,data_doses_rate)])
	p1_rate = [s1_rate, i1_rate]
	p1_failed = [1, 1.e-5, 1]
	p1_false = [1, 1.e-5, 1]
	coeff_d_rate = curve_fit(linear, doses, data_doses_rate, p0 = p1_rate, full_output=True)
	coeff_d_failed = curve_fit(expdec, doses, data_doses_failed, p0 = p1_failed, full_output=True)
	coeff_d_false = curve_fit(expdec,  doses, data_doses_false, p0 = p1_false, full_output=True)
	
	return coeff_d_rate, coeff_d_failed, coeff_d_false

def populate_prediction_data(predicted_doses,sevs,seeds):
	rate_predictor_00sst = []
	rate_predictor_10sst = []
	rate_predictor_20sst = []
	rate_predictor_30sst = []
	rate_predictor_40sst = []
	failed_predictor_00sst = []
	failed_predictor_10sst = []
	failed_predictor_20sst = []
	failed_predictor_30sst = []
	failed_predictor_40sst = []
	false_predictor_00sst = []
	false_predictor_10sst = []
	false_predictor_20sst = []
	false_predictor_30sst = []
	false_predictor_40sst = []
	for ind, predicted_dose in enumerate(predicted_doses):
		
		cind2 = np.where(severities == sevs[ind])[0][0]
		seedind = seeds[ind]-1

		if cind2 == 0:
			coeff_d_rate, coeff_d_failed, coeff_d_false = get_dose_response_fits(cind2,seedind)
			zerodose_r = base_rates[0][cind2][seedind]
			preddose_r = linear(predicted_doses[ind],*coeff_d_rate[0])
			zerodose_fai = failed_detections[0][cind2][seedind]
			preddose_fai = expdec(predicted_doses[ind],*coeff_d_failed[0])
			zerodose_fal = false_detections[0][cind2][seedind]
			preddose_fal = expdec(predicted_doses[ind],*coeff_d_false[0])
			
			rate_predictor_00sst.append([zerodose_r,preddose_r])
			failed_predictor_00sst.append([zerodose_fai,preddose_fai])
			false_predictor_00sst.append([zerodose_fal,preddose_fal])
		elif cind2 == 1:
			coeff_d_rate, coeff_d_failed, coeff_d_false = get_dose_response_fits(cind2,seedind)
			zerodose_r = base_rates[0][cind2][seedind]
			preddose_r = linear(predicted_doses[ind],*coeff_d_rate[0])
			zerodose_fai = failed_detections[0][cind2][seedind]
			preddose_fai = expdec(predicted_doses[ind],*coeff_d_failed[0])
			zerodose_fal = false_detections[0][cind2][seedind]
			preddose_fal = expdec(predicted_doses[ind],*coeff_d_false[0])
			
			rate_predictor_10sst.append([zerodose_r,preddose_r])
			failed_predictor_10sst.append([zerodose_fai,preddose_fai])
			false_predictor_10sst.append([zerodose_fal,preddose_fal])
		elif cind2 == 2:
			coeff_d_rate, coeff_d_failed, coeff_d_false = get_dose_response_fits(cind2,seedind)
			zerodose_r = base_rates[0][cind2][seedind]
			preddose_r = linear(predicted_doses[ind],*coeff_d_rate[0])
			zerodose_fai = failed_detections[0][cind2][seedind]
			preddose_fai = expdec(predicted_doses[ind],*coeff_d_failed[0])
			zerodose_fal = false_detections[0][cind2][seedind]
			preddose_fal = expdec(predicted_doses[ind],*coeff_d_false[0])
			
			rate_predictor_20sst.append([zerodose_r,preddose_r])
			failed_predictor_20sst.append([zerodose_fai,preddose_fai])
			false_predictor_20sst.append([zerodose_fal,preddose_fal])
		elif cind2 == 3:
			coeff_d_rate, coeff_d_failed, coeff_d_false = get_dose_response_fits(cind2,seedind)
			zerodose_r = base_rates[0][cind2][seedind]
			preddose_r = linear(predicted_doses[ind],*coeff_d_rate[0])
			zerodose_fai = failed_detections[0][cind2][seedind]
			preddose_fai = expdec(predicted_doses[ind],*coeff_d_failed[0])
			zerodose_fal = false_detections[0][cind2][seedind]
			preddose_fal = expdec(predicted_doses[ind],*coeff_d_false[0])
			
			rate_predictor_30sst.append([zerodose_r,preddose_r])
			failed_predictor_30sst.append([zerodose_fai,preddose_fai])
			false_predictor_30sst.append([zerodose_fal,preddose_fal])
		elif cind2 == 4:
			coeff_d_rate, coeff_d_failed, coeff_d_false = get_dose_response_fits(cind2,seedind)
			zerodose_r = base_rates[0][cind2][seedind]
			preddose_r = linear(predicted_doses[ind],*coeff_d_rate[0])
			zerodose_fai = failed_detections[0][cind2][seedind]
			preddose_fai = expdec(predicted_doses[ind],*coeff_d_failed[0])
			zerodose_fal = false_detections[0][cind2][seedind]
			preddose_fal = expdec(predicted_doses[ind],*coeff_d_false[0])
			
			rate_predictor_40sst.append([zerodose_r,preddose_r])
			failed_predictor_40sst.append([zerodose_fai,preddose_fai])
			false_predictor_40sst.append([zerodose_fal,preddose_fal])

	return rate_predictor_00sst, rate_predictor_10sst, rate_predictor_20sst, rate_predictor_30sst, rate_predictor_40sst, failed_predictor_00sst, failed_predictor_10sst, failed_predictor_20sst, failed_predictor_30sst, failed_predictor_40sst, false_predictor_00sst, false_predictor_10sst, false_predictor_20sst, false_predictor_30sst, false_predictor_40sst

rate_mv_00sst,rate_mv_10sst,rate_mv_20sst,rate_mv_30sst,rate_mv_40sst,failed_mv_00sst,failed_mv_10sst,failed_mv_20sst,failed_mv_30sst,failed_mv_40sst,false_mv_00sst,false_mv_10sst,false_mv_20sst,false_mv_30sst,false_mv_40sst = populate_prediction_data(test,best_test0[4],best_test0[5])
rate_aperiodic_00sst,rate_aperiodic_10sst,rate_aperiodic_20sst,rate_aperiodic_30sst,rate_aperiodic_40sst,failed_aperiodic_00sst,failed_aperiodic_10sst,failed_aperiodic_20sst,failed_aperiodic_30sst,failed_aperiodic_40sst,false_aperiodic_00sst,false_aperiodic_10sst,false_aperiodic_20sst,false_aperiodic_30sst,false_aperiodic_40sst = populate_prediction_data(test_linear0,best_test1[4],best_test1[5])
rate_theta_00sst,rate_theta_10sst,rate_theta_20sst,rate_theta_30sst,rate_theta_40sst,failed_theta_00sst,failed_theta_10sst,failed_theta_20sst,failed_theta_30sst,failed_theta_40sst,false_theta_00sst,false_theta_10sst,false_theta_20sst,false_theta_30sst,false_theta_40sst = populate_prediction_data(test_linear1,best_test2[4],best_test2[5])
rate_alpha_00sst,rate_alpha_10sst,rate_alpha_20sst,rate_alpha_30sst,rate_alpha_40sst,failed_alpha_00sst,failed_alpha_10sst,failed_alpha_20sst,failed_alpha_30sst,failed_alpha_40sst,false_alpha_00sst,false_alpha_10sst,false_alpha_20sst,false_alpha_30sst,false_alpha_40sst = populate_prediction_data(test_linear2,best_test3[4],best_test3[5])
rate_rate_00sst,rate_rate_10sst,rate_rate_20sst,rate_rate_30sst,rate_rate_40sst,failed_rate_00sst,failed_rate_10sst,failed_rate_20sst,failed_rate_30sst,failed_rate_40sst,false_rate_00sst,false_rate_10sst,false_rate_20sst,false_rate_30sst,false_rate_40sst = populate_prediction_data(test_linear3,best_test4[4],best_test4[5])

# Now for EEG biomarker metrics
def get_dose_response_fits_EEG(cind2_,seedind_):
	data_doses_aperiodic =[x[cind2_][seedind_] for x in AUC]
	data_doses_theta =[x[cind2_][seedind_] for x in AUC_t_abs]
	data_doses_alpha =[x[cind2_][seedind_] for x in AUC_a_abs]
	s1_aperiodic = (doses[-1]-doses[0])/(data_doses_aperiodic[-1]-data_doses_aperiodic[0])
	s1_theta = (doses[-1]-doses[0])/(data_doses_theta[-1]-data_doses_theta[0])
	s1_alpha = (doses[-1]-doses[0])/(data_doses_alpha[-1]-data_doses_alpha[0])
	i1_aperiodic = np.mean([od - s1_aperiodic*d1 for od,d1 in zip(doses,data_doses_aperiodic)])
	i1_theta = np.mean([od - s1_theta*d1 for od,d1 in zip(doses,data_doses_theta)])
	i1_alpha = np.mean([od - s1_alpha*d1 for od,d1 in zip(doses,data_doses_alpha)])
	p1_aperiodic = [s1_aperiodic, i1_aperiodic]
	p1_theta = [s1_theta, i1_theta]
	p1_alpha = [s1_alpha, i1_alpha]
	coeff_d_aperiodic = curve_fit(linear, doses, data_doses_aperiodic, p0 = p1_aperiodic, full_output=True)
	coeff_d_theta = curve_fit(linear, doses, data_doses_theta, p0 = p1_theta, full_output=True)
	coeff_d_alpha = curve_fit(linear,  doses, data_doses_alpha, p0 = p1_alpha, full_output=True)
	
	return coeff_d_aperiodic, coeff_d_theta, coeff_d_alpha

def populate_prediction_data_EEG(predicted_doses,sevs,seeds):
	aperiodic_predictor_00sst = []
	aperiodic_predictor_10sst = []
	aperiodic_predictor_20sst = []
	aperiodic_predictor_30sst = []
	aperiodic_predictor_40sst = []
	theta_predictor_00sst = []
	theta_predictor_10sst = []
	theta_predictor_20sst = []
	theta_predictor_30sst = []
	theta_predictor_40sst = []
	alpha_predictor_00sst = []
	alpha_predictor_10sst = []
	alpha_predictor_20sst = []
	alpha_predictor_30sst = []
	alpha_predictor_40sst = []
	for ind, predicted_dose in enumerate(predicted_doses):
		
		cind2 = np.where(severities == sevs[ind])[0][0]
		seedind = seeds[ind]-1
		
		if cind2 == 0:
			coeff_d_aperiodic, coeff_d_theta, coeff_d_alpha = get_dose_response_fits_EEG(cind2,seedind)
			zerodose_ap = AUC[0][cind2][seedind]
			preddose_ap = linear(predicted_doses[ind],*coeff_d_aperiodic[0])
			zerodose_th = AUC_t_abs[0][cind2][seedind]
			preddose_th = linear(predicted_doses[ind],*coeff_d_theta[0])
			zerodose_al = AUC_a_abs[0][cind2][seedind]
			preddose_al = linear(predicted_doses[ind],*coeff_d_alpha[0])
			
			aperiodic_predictor_00sst.append([zerodose_ap,preddose_ap])
			theta_predictor_00sst.append([zerodose_th,preddose_th])
			alpha_predictor_00sst.append([zerodose_al,preddose_al])
		elif cind2 == 1:
			coeff_d_aperiodic, coeff_d_theta, coeff_d_alpha = get_dose_response_fits_EEG(cind2,seedind)
			zerodose_ap = AUC[0][cind2][seedind]
			preddose_ap = linear(predicted_doses[ind],*coeff_d_aperiodic[0])
			zerodose_th = AUC_t_abs[0][cind2][seedind]
			preddose_th = linear(predicted_doses[ind],*coeff_d_theta[0])
			zerodose_al = AUC_a_abs[0][cind2][seedind]
			preddose_al = linear(predicted_doses[ind],*coeff_d_alpha[0])
			
			aperiodic_predictor_10sst.append([zerodose_ap,preddose_ap])
			theta_predictor_10sst.append([zerodose_th,preddose_th])
			alpha_predictor_10sst.append([zerodose_al,preddose_al])
		elif cind2 == 2:
			coeff_d_aperiodic, coeff_d_theta, coeff_d_alpha = get_dose_response_fits_EEG(cind2,seedind)
			zerodose_ap = AUC[0][cind2][seedind]
			preddose_ap = linear(predicted_doses[ind],*coeff_d_aperiodic[0])
			zerodose_th = AUC_t_abs[0][cind2][seedind]
			preddose_th = linear(predicted_doses[ind],*coeff_d_theta[0])
			zerodose_al = AUC_a_abs[0][cind2][seedind]
			preddose_al = linear(predicted_doses[ind],*coeff_d_alpha[0])
			
			aperiodic_predictor_20sst.append([zerodose_ap,preddose_ap])
			theta_predictor_20sst.append([zerodose_th,preddose_th])
			alpha_predictor_20sst.append([zerodose_al,preddose_al])
		elif cind2 == 3:
			coeff_d_aperiodic, coeff_d_theta, coeff_d_alpha = get_dose_response_fits_EEG(cind2,seedind)
			zerodose_ap = AUC[0][cind2][seedind]
			preddose_ap = linear(predicted_doses[ind],*coeff_d_aperiodic[0])
			zerodose_th = AUC_t_abs[0][cind2][seedind]
			preddose_th = linear(predicted_doses[ind],*coeff_d_theta[0])
			zerodose_al = AUC_a_abs[0][cind2][seedind]
			preddose_al = linear(predicted_doses[ind],*coeff_d_alpha[0])
			
			aperiodic_predictor_30sst.append([zerodose_ap,preddose_ap])
			theta_predictor_30sst.append([zerodose_th,preddose_th])
			alpha_predictor_30sst.append([zerodose_al,preddose_al])
		elif cind2 == 4:
			coeff_d_aperiodic, coeff_d_theta, coeff_d_alpha = get_dose_response_fits_EEG(cind2,seedind)
			zerodose_ap = AUC[0][cind2][seedind]
			preddose_ap = linear(predicted_doses[ind],*coeff_d_aperiodic[0])
			zerodose_th = AUC_t_abs[0][cind2][seedind]
			preddose_th = linear(predicted_doses[ind],*coeff_d_theta[0])
			zerodose_al = AUC_a_abs[0][cind2][seedind]
			preddose_al = linear(predicted_doses[ind],*coeff_d_alpha[0])

			aperiodic_predictor_40sst.append([zerodose_ap,preddose_ap])
			theta_predictor_40sst.append([zerodose_th,preddose_th])
			alpha_predictor_40sst.append([zerodose_al,preddose_al])

	return aperiodic_predictor_00sst, aperiodic_predictor_10sst, aperiodic_predictor_20sst, aperiodic_predictor_30sst, aperiodic_predictor_40sst, theta_predictor_00sst, theta_predictor_10sst, theta_predictor_20sst, theta_predictor_30sst, theta_predictor_40sst, alpha_predictor_00sst, alpha_predictor_10sst, alpha_predictor_20sst, alpha_predictor_30sst, alpha_predictor_40sst

aperiodic_mv_00sst,aperiodic_mv_10sst,aperiodic_mv_20sst,aperiodic_mv_30sst,aperiodic_mv_40sst,theta_mv_00sst,theta_mv_10sst,theta_mv_20sst,theta_mv_30sst,theta_mv_40sst,alpha_mv_00sst,alpha_mv_10sst,alpha_mv_20sst,alpha_mv_30sst,alpha_mv_40sst = populate_prediction_data_EEG(test,best_test0[4],best_test0[5])
aperiodic_rate_00sst,aperiodic_rate_10sst,aperiodic_rate_20sst,aperiodic_rate_30sst,aperiodic_rate_40sst,theta_rate_00sst,theta_rate_10sst,theta_rate_20sst,theta_rate_30sst,theta_rate_40sst,alpha_rate_00sst,alpha_rate_10sst,alpha_rate_20sst,alpha_rate_30sst,alpha_rate_40sst = populate_prediction_data_EEG(test_linear3,best_test4[4],best_test4[5])

### Add plots showing 3D space (maybe do the same for EEG biomarker values)
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')

x_lims = failed_detections_lu[0][0]
y_lims = false_detections_lu[0][0]
z_lims = base_rates_lu[0][0]
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

combined_failed_mv = failed_mv_10sst + failed_mv_20sst + failed_mv_30sst + failed_mv_40sst
combined_false_mv = false_mv_10sst + false_mv_20sst + false_mv_30sst + false_mv_40sst
combined_rate_mv = rate_mv_10sst + rate_mv_20sst + rate_mv_30sst + rate_mv_40sst

minmax_x = [999,0]
minmax_y = [999,0]
minmax_z = [999,0]
for m1,m2,m3 in zip(combined_false_mv,combined_false_mv,combined_rate_mv):
	if np.min(m1) < minmax_x[0]: minmax_x[0] = np.min(m1)
	if np.max(m1) > minmax_x[1]: minmax_x[1] = np.max(m1)
	if np.min(m2) < minmax_y[0]: minmax_y[0] = np.min(m2)
	if np.max(m2) > minmax_y[1]: minmax_y[1] = np.max(m2)
	if np.min(m3) < minmax_z[0]: minmax_z[0] = np.min(m3)
	if np.max(m3) > minmax_z[1]: minmax_z[1] = np.max(m3)
	ax.scatter(m1[0], m2[0], m3[0], c='white', s=14**2, edgecolors='magenta', linewidth=2) # Plot magenta color to outline zero dose
	ax.scatter(m1[1], m2[1], m3[1], c='white', s=14**2, edgecolors='dodgerblue', linewidth=2) # Plot magenta color to outline zero dose
ax.set_xlabel('\n\n Failed (%)')
ax.set_ylabel('\n\n False (%)')
ax.set_zlabel('\n\n Rate (Hz)')
ax.set_xlim(minmax_x[0]-1,minmax_x[1]+1)
ax.set_ylim(minmax_y[0]-1,minmax_y[1]+1)
ax.set_zlim(minmax_z[0]-0.1,minmax_z[1]+0.1)
#ax.view_init(90, -90)
ax.xaxis.set_major_locator(mticker.MultipleLocator(2)) # disable if using smaller axis values <1
ax.yaxis.set_major_locator(mticker.MultipleLocator(2)) # disable if using smaller axis values <1
ax.zaxis.set_major_locator(mticker.MultipleLocator(0.3)) # disable if using smaller axis values <1
fig.tight_layout()

fig.savefig('figs_ManualLinearRegression_PSDvalidation_V5_ANN/opt_dose_fit_severity_3D_performance_MVpredicted.png',dpi=300,transparent=True)
plt.close()

# Now for EEG biomarker metric recovery
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')

x_lims = AUC_lu[0][0]
y_lims = AUC_t_abs_lu[0][0]
z_lims = AUC_a_abs_lu[0][0]
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

combined_aperiodic_mv = aperiodic_mv_10sst + aperiodic_mv_20sst + aperiodic_mv_30sst + aperiodic_mv_40sst
combined_theta_mv = theta_mv_10sst + theta_mv_20sst + theta_mv_30sst + theta_mv_40sst
combined_alpha_mv = alpha_mv_10sst + alpha_mv_20sst + alpha_mv_30sst + alpha_mv_40sst

minmax_x = [999,0]
minmax_y = [999,0]
minmax_z = [999,0]
count_within_healthy = 0
for m1,m2,m3 in zip(combined_aperiodic_mv,combined_theta_mv,combined_alpha_mv):
	if np.min(m1) < minmax_x[0]: minmax_x[0] = np.min(m1)
	if np.max(m1) > minmax_x[1]: minmax_x[1] = np.max(m1)
	if np.min(m2) < minmax_y[0]: minmax_y[0] = np.min(m2)
	if np.max(m2) > minmax_y[1]: minmax_y[1] = np.max(m2)
	if np.min(m3) < minmax_z[0]: minmax_z[0] = np.min(m3)
	if np.max(m3) > minmax_z[1]: minmax_z[1] = np.max(m3)
	if (x_lims[0] <= m1[1] <= x_lims[1]) and (y_lims[0] <= m2[1] <= y_lims[1]) and (z_lims[0] <= m3[1] <= z_lims[1]):
		count_within_healthy += 1
	ax.scatter(m1[0], m2[0], m3[0], c='magenta', s=14**2, edgecolors='k', linewidth=2) # Plot magenta color to outline zero dose
	ax.scatter(m1[1], m2[1], m3[1], c='dodgerblue', s=14**2, edgecolors='k', linewidth=2) # Plot magenta color to outline zero dose

print('Fraction of subjects with recovered biomarkers: ' + str((count_within_healthy/len(combined_aperiodic_mv))*100))

print(count_within_healthy)
print(len(combined_aperiodic_mv))

ax.set_xlabel('\n\n1/f')
ax.set_ylabel('\n\n'+r'$\theta$')
ax.set_zlabel('\n'+r'$\alpha$')
ax.set_xlim(minmax_x[0]-0.1,minmax_x[1]+0.1)
ax.set_ylim(minmax_y[0]-0.1,minmax_y[1]+0.1)
ax.set_zlim(minmax_z[0]-0.1,minmax_z[1]+0.1)
#ax.view_init(90, -90)
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.3)) # disable if using smaller axis values <1
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.3)) # disable if using smaller axis values <1
ax.zaxis.set_major_locator(mticker.MultipleLocator(0.3)) # disable if using smaller axis values <1
fig.tight_layout()

fig.savefig('figs_ManualLinearRegression_PSDvalidation_V5_ANN/opt_dose_fit_severity_3D_biomarkers_MVpredicted.png',dpi=300,transparent=True)
plt.close()

# Rate-predicted one now
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')

x_lims = AUC_lu[0][0]
y_lims = AUC_t_abs_lu[0][0]
z_lims = AUC_a_abs_lu[0][0]
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

combined_failed_rate = failed_rate_10sst + failed_rate_20sst + failed_rate_30sst + failed_rate_40sst
combined_false_rate = false_rate_10sst + false_rate_20sst + false_rate_30sst + false_rate_40sst
combined_rate_rate = rate_rate_10sst + rate_rate_20sst + rate_rate_30sst + rate_rate_40sst

combined_aperiodic_rate = aperiodic_rate_10sst + aperiodic_rate_20sst + aperiodic_rate_30sst + aperiodic_rate_40sst
combined_theta_rate = theta_rate_10sst + theta_rate_20sst + theta_rate_30sst + theta_rate_40sst
combined_alpha_rate = alpha_rate_10sst + alpha_rate_20sst + alpha_rate_30sst + alpha_rate_40sst

minmax_x = [999,0]
minmax_y = [999,0]
minmax_z = [999,0]
count_within_healthy = 0
for m1,m2,m3 in zip(combined_aperiodic_rate,combined_theta_rate,combined_alpha_rate):
	if np.min(m1) < minmax_x[0]: minmax_x[0] = np.min(m1)
	if np.max(m1) > minmax_x[1]: minmax_x[1] = np.max(m1)
	if np.min(m2) < minmax_y[0]: minmax_y[0] = np.min(m2)
	if np.max(m2) > minmax_y[1]: minmax_y[1] = np.max(m2)
	if np.min(m3) < minmax_z[0]: minmax_z[0] = np.min(m3)
	if np.max(m3) > minmax_z[1]: minmax_z[1] = np.max(m3)
	if (x_lims[0] <= m1[1] <= x_lims[1]) and (y_lims[0] <= m2[1] <= y_lims[1]) and (z_lims[0] <= m3[1] <= z_lims[1]):
		count_within_healthy += 1
	ax.scatter(m1[0], m2[0], m3[0], c='magenta', s=14**2, edgecolors='k', linewidth=2) # Plot magenta color to outline zero dose
	ax.scatter(m1[1], m2[1], m3[1], c='dodgerblue', s=14**2, edgecolors='k', linewidth=2) # Plot magenta color to outline zero dose

print('Fraction of subjects with recovered biomarkers (rate-predicted doses): ' + str((count_within_healthy/len(combined_aperiodic_rate))*100))

print(count_within_healthy)
print(len(combined_aperiodic_rate))

ax.set_xlabel('\n\n1/f')
ax.set_ylabel('\n\n'+r'$\theta$')
ax.set_zlabel('\n'+r'$\alpha$')
ax.set_xlim(minmax_x[0]-0.1,minmax_x[1]+0.1)
ax.set_ylim(minmax_y[0]-0.1,minmax_y[1]+0.1)
ax.set_zlim(minmax_z[0]-0.1,minmax_z[1]+0.1)
#ax.view_init(90, -90)
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.3)) # disable if using smaller axis values <1
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.3)) # disable if using smaller axis values <1
ax.zaxis.set_major_locator(mticker.MultipleLocator(0.3)) # disable if using smaller axis values <1
fig.tight_layout()

fig.savefig('figs_ManualLinearRegression_PSDvalidation_V5_ANN/opt_dose_fit_severity_3D_biomarkers_RATEpredicted.png',dpi=300,transparent=True)
plt.close()

# Stats compared to healthy
c_rate_mv = [c[1] for c in combined_rate_mv]
c_failed_mv = [c[1] for c in combined_failed_mv]
c_false_mv = [c[1] for c in combined_false_mv]
c_aperiodic_mv = [c[1] for c in combined_aperiodic_mv]
c_theta_mv = [c[1] for c in combined_theta_mv]
c_alpha_mv = [c[1] for c in combined_alpha_mv]

c_rate_rate = [c[1] for c in combined_rate_rate]
c_failed_rate = [c[1] for c in combined_failed_rate]
c_false_rate = [c[1] for c in combined_false_rate]
c_aperiodic_rate = [c[1] for c in combined_aperiodic_rate]
c_theta_rate = [c[1] for c in combined_theta_rate]
c_alpha_rate = [c[1] for c in combined_alpha_rate]

metric_names = ["Rate","Failed","False","1/f","Theta","Alpha"]
mean_rate_h = np.mean(base_rates[0][0])
mean_failed_h = np.mean(failed_detections[0][0])
mean_false_h = np.mean(false_detections[0][0])
mean_aperiodic_h = np.mean(AUC[0][0])
mean_theta_h = np.mean(AUC_t_abs[0][0])
mean_alpha_h = np.mean(AUC_a_abs[0][0])
means_h = [mean_rate_h,mean_failed_h,mean_false_h,mean_aperiodic_h,mean_theta_h,mean_alpha_h]

mean_rate_p = np.mean(c_rate_mv)
mean_failed_p = np.mean(c_failed_mv)
mean_false_p = np.mean(c_false_mv)
mean_aperiodic_p = np.mean(c_aperiodic_mv)
mean_theta_p = np.mean(c_theta_mv)
mean_alpha_p = np.mean(c_alpha_mv)
means_p = [mean_rate_p,mean_failed_p,mean_false_p,mean_aperiodic_p,mean_theta_p,mean_alpha_p]

mean_rate_pr = np.mean(c_rate_rate)
mean_failed_pr = np.mean(c_failed_rate)
mean_false_pr = np.mean(c_false_rate)
mean_aperiodic_pr = np.mean(c_aperiodic_rate)
mean_theta_pr = np.mean(c_theta_rate)
mean_alpha_pr = np.mean(c_alpha_rate)
means_pr = [mean_rate_pr,mean_failed_pr,mean_false_pr,mean_aperiodic_pr,mean_theta_pr,mean_alpha_pr]

sd_rate_h = np.std(base_rates[0][0])
sd_failed_h = np.std(failed_detections[0][0])
sd_false_h = np.std(false_detections[0][0])
sd_aperiodic_h = np.std(AUC[0][0])
sd_theta_h = np.std(AUC_t_abs[0][0])
sd_alpha_h = np.std(AUC_a_abs[0][0])
sd_h = [sd_rate_h,sd_failed_h,sd_false_h,sd_aperiodic_h,sd_theta_h,sd_alpha_h]

sd_rate_p = np.std(c_rate_mv)
sd_failed_p = np.std(c_failed_mv)
sd_false_p = np.std(c_false_mv)
sd_aperiodic_p = np.std(c_aperiodic_mv)
sd_theta_p = np.std(c_theta_mv)
sd_alpha_p = np.std(c_alpha_mv)
sd_p = [sd_rate_p,sd_failed_p,sd_false_p,sd_aperiodic_p,sd_theta_p,sd_alpha_p]

sd_rate_pr = np.std(c_rate_rate)
sd_failed_pr = np.std(c_failed_rate)
sd_false_pr = np.std(c_false_rate)
sd_aperiodic_pr = np.std(c_aperiodic_rate)
sd_theta_pr = np.std(c_theta_rate)
sd_alpha_pr = np.std(c_alpha_rate)
sd_pr = [sd_rate_pr,sd_failed_pr,sd_false_pr,sd_aperiodic_pr,sd_theta_pr,sd_alpha_pr]

normal_stat_rate_h, normal_p_rate_h = st.normaltest(base_rates[0][0])
normal_stat_failed_h, normal_p_failed_h = st.normaltest(failed_detections[0][0])
normal_stat_false_h, normal_p_false_h = st.normaltest(false_detections[0][0])
normal_stat_aperiodic_h, normal_p_aperiodic_h = st.normaltest(AUC[0][0])
normal_stat_theta_h, normal_p_theta_h = st.normaltest(AUC_t_abs[0][0])
normal_stat_alpha_h, normal_p_alpha_h = st.normaltest(AUC_a_abs[0][0])
normal_stat_h = [normal_stat_rate_h,normal_stat_failed_h,normal_stat_false_h,normal_stat_aperiodic_h,normal_stat_theta_h,normal_stat_alpha_h]
normal_pval_h = [normal_p_rate_h,normal_p_failed_h,normal_p_false_h,normal_p_aperiodic_h,normal_p_theta_h,normal_p_alpha_h]

normal_stat_rate_p, normal_p_rate_p = st.normaltest(c_rate_mv)
normal_stat_failed_p, normal_p_failed_p = st.normaltest(c_failed_mv)
normal_stat_false_p, normal_p_false_p = st.normaltest(c_false_mv)
normal_stat_aperiodic_p, normal_p_aperiodic_p = st.normaltest(c_aperiodic_mv)
normal_stat_theta_p, normal_p_theta_p = st.normaltest(c_theta_mv)
normal_stat_alpha_p, normal_p_alpha_p = st.normaltest(c_alpha_mv)
normal_stat_p = [normal_stat_rate_p,normal_stat_failed_p,normal_stat_false_p,normal_stat_aperiodic_p,normal_stat_theta_p,normal_stat_alpha_p]
normal_pval_p = [normal_p_rate_p,normal_p_failed_p,normal_p_false_p,normal_p_aperiodic_p,normal_p_theta_p,normal_p_alpha_p]

normal_stat_rate_pr, normal_p_rate_pr = st.normaltest(c_rate_rate)
normal_stat_failed_pr, normal_p_failed_pr = st.normaltest(c_failed_rate)
normal_stat_false_pr, normal_p_false_pr = st.normaltest(c_false_rate)
normal_stat_aperiodic_pr, normal_p_aperiodic_pr = st.normaltest(c_aperiodic_rate)
normal_stat_theta_pr, normal_p_theta_pr = st.normaltest(c_theta_rate)
normal_stat_alpha_pr, normal_p_alpha_pr = st.normaltest(c_alpha_rate)
normal_stat_pr = [normal_stat_rate_pr,normal_stat_failed_pr,normal_stat_false_pr,normal_stat_aperiodic_pr,normal_stat_theta_pr,normal_stat_alpha_pr]
normal_pval_pr = [normal_p_rate_pr,normal_p_failed_pr,normal_p_false_pr,normal_p_aperiodic_pr,normal_p_theta_pr,normal_p_alpha_pr]

levene_stat_rate, p_levene_rate = st.levene(base_rates[0][0],c_rate_mv)
levene_stat_failed, p_levene_failed = st.levene(failed_detections[0][0],c_failed_mv)
levene_stat_false, p_levene_false = st.levene(false_detections[0][0],c_false_mv)
levene_stat_aperiodic, p_levene_aperiodic = st.levene(AUC[0][0],c_aperiodic_mv)
levene_stat_theta, p_levene_theta = st.levene(AUC_t_abs[0][0],c_theta_mv)
levene_stat_alpha, p_levene_alpha = st.levene(AUC_a_abs[0][0],c_alpha_mv)
levene_stat = [levene_stat_rate,levene_stat_failed,levene_stat_false,levene_stat_aperiodic,levene_stat_theta,levene_stat_alpha]
levene_pval = [p_levene_rate,p_levene_failed,p_levene_false,p_levene_aperiodic,p_levene_theta,p_levene_alpha]

levene_stat_rater, p_levene_rater = st.levene(base_rates[0][0],c_rate_rate)
levene_stat_failedr, p_levene_failedr = st.levene(failed_detections[0][0],c_failed_rate)
levene_stat_falser, p_levene_falser = st.levene(false_detections[0][0],c_false_rate)
levene_stat_aperiodicr, p_levene_aperiodicr = st.levene(AUC[0][0],c_aperiodic_rate)
levene_stat_thetar, p_levene_thetar = st.levene(AUC_t_abs[0][0],c_theta_rate)
levene_stat_alphar, p_levene_alphar = st.levene(AUC_a_abs[0][0],c_alpha_rate)
levene_statr = [levene_stat_rater,levene_stat_failedr,levene_stat_falser,levene_stat_aperiodicr,levene_stat_thetar,levene_stat_alphar]
levene_pvalr = [p_levene_rater,p_levene_failedr,p_levene_falser,p_levene_aperiodicr,p_levene_thetar,p_levene_alphar]

ttest_stat_rate, p_ttest_rate = st.ttest_ind(base_rates[0][0],c_rate_mv,equal_var=True if p_levene_rate >= 0.05 else False)
ttest_stat_failed, p_ttest_failed = st.ttest_ind(failed_detections[0][0],c_failed_mv,equal_var=True if p_levene_failed >= 0.05 else False)
ttest_stat_false, p_ttest_false = st.ttest_ind(false_detections[0][0],c_false_mv,equal_var=True if p_levene_false >= 0.05 else False)
ttest_stat_aperiodic, p_ttest_aperiodic = st.ttest_ind(AUC[0][0],c_aperiodic_mv,equal_var=True if p_levene_aperiodic >= 0.05 else False)
ttest_stat_theta, p_ttest_theta = st.ttest_ind(AUC_t_abs[0][0],c_theta_mv,equal_var=True if p_levene_theta >= 0.05 else False)
ttest_stat_alpha, p_ttest_alpha = st.ttest_ind(AUC_a_abs[0][0],c_alpha_mv,equal_var=True if p_levene_alpha >= 0.05 else False)
ttest_stat = [ttest_stat_rate,ttest_stat_failed,ttest_stat_false,ttest_stat_aperiodic,ttest_stat_theta,ttest_stat_alpha]
ttest_pval = [p_ttest_rate,p_ttest_failed,p_ttest_false,p_ttest_aperiodic,p_ttest_theta,p_ttest_alpha]

ttest_stat_rater, p_ttest_rater = st.ttest_ind(base_rates[0][0],c_rate_rate,equal_var=True if p_levene_rater >= 0.05 else False)
ttest_stat_failedr, p_ttest_failedr = st.ttest_ind(failed_detections[0][0],c_failed_rate,equal_var=True if p_levene_failedr >= 0.05 else False)
ttest_stat_falser, p_ttest_falser = st.ttest_ind(false_detections[0][0],c_false_rate,equal_var=True if p_levene_falser >= 0.05 else False)
ttest_stat_aperiodicr, p_ttest_aperiodicr = st.ttest_ind(AUC[0][0],c_aperiodic_rate,equal_var=True if p_levene_aperiodicr >= 0.05 else False)
ttest_stat_thetar, p_ttest_thetar = st.ttest_ind(AUC_t_abs[0][0],c_theta_rate,equal_var=True if p_levene_thetar >= 0.05 else False)
ttest_stat_alphar, p_ttest_alphar = st.ttest_ind(AUC_a_abs[0][0],c_alpha_rate,equal_var=True if p_levene_alphar >= 0.05 else False)
ttest_statr = [ttest_stat_rater,ttest_stat_failedr,ttest_stat_falser,ttest_stat_aperiodicr,ttest_stat_thetar,ttest_stat_alphar]
ttest_pvalr = [p_ttest_rater,p_ttest_failedr,p_ttest_falser,p_ttest_aperiodicr,p_ttest_thetar,p_ttest_alphar]

mw_stat_rate, p_mw_rate = st.mannwhitneyu(base_rates[0][0],c_rate_mv)
mw_stat_failed, p_mw_failed = st.mannwhitneyu(failed_detections[0][0],c_failed_mv)
mw_stat_false, p_mw_false = st.mannwhitneyu(false_detections[0][0],c_false_mv)
mw_stat_aperiodic, p_mw_aperiodic = st.mannwhitneyu(AUC[0][0],c_aperiodic_mv)
mw_stat_theta, p_mw_theta = st.mannwhitneyu(AUC_t_abs[0][0],c_theta_mv)
mw_stat_alpha, p_mw_alpha = st.mannwhitneyu(AUC_a_abs[0][0],c_alpha_mv)
mw_stat = [mw_stat_rate,mw_stat_failed,mw_stat_false,mw_stat_aperiodic,mw_stat_theta,mw_stat_alpha]
mw_pval = [p_mw_rate,p_mw_failed,p_mw_false,p_mw_aperiodic,p_mw_theta,p_mw_alpha]

mw_stat_rater, p_mw_rater = st.mannwhitneyu(base_rates[0][0],c_rate_rate)
mw_stat_failedr, p_mw_failedr = st.mannwhitneyu(failed_detections[0][0],c_failed_rate)
mw_stat_falser, p_mw_falser = st.mannwhitneyu(false_detections[0][0],c_false_rate)
mw_stat_aperiodicr, p_mw_aperiodicr = st.mannwhitneyu(AUC[0][0],c_aperiodic_rate)
mw_stat_thetar, p_mw_thetar = st.mannwhitneyu(AUC_t_abs[0][0],c_theta_rate)
mw_stat_alphar, p_mw_alphar = st.mannwhitneyu(AUC_a_abs[0][0],c_alpha_rate)
mw_statr = [mw_stat_rater,mw_stat_failedr,mw_stat_falser,mw_stat_aperiodicr,mw_stat_thetar,mw_stat_alphar]
mw_pvalr = [p_mw_rater,p_mw_failedr,p_mw_falser,p_mw_aperiodicr,p_mw_thetar,p_mw_alphar]

cd_rate = cohen_d(base_rates[0][0],c_rate_mv)
cd_failed = cohen_d(failed_detections[0][0],c_failed_mv)
cd_false = cohen_d(false_detections[0][0],c_false_mv)
cd_aperiodic = cohen_d(AUC[0][0],c_aperiodic_mv)
cd_theta = cohen_d(AUC_t_abs[0][0],c_theta_mv)
cd_alpha = cohen_d(AUC_a_abs[0][0],c_alpha_mv)
cds = [cd_rate,cd_failed,cd_false,cd_aperiodic,cd_theta,cd_alpha]

cd_rater = cohen_d(base_rates[0][0],c_rate_rate)
cd_failedr = cohen_d(failed_detections[0][0],c_failed_rate)
cd_falser = cohen_d(false_detections[0][0],c_false_rate)
cd_aperiodicr = cohen_d(AUC[0][0],c_aperiodic_rate)
cd_thetar = cohen_d(AUC_t_abs[0][0],c_theta_rate)
cd_alphar = cohen_d(AUC_a_abs[0][0],c_alpha_rate)
cdsr = [cd_rater,cd_failedr,cd_falser,cd_aperiodicr,cd_thetar,cd_alphar]

allstats = zip(metric_names,
				means_h,
				means_p,
				sd_h,
				sd_p,
				normal_stat_h,
				normal_pval_h,
				normal_stat_p,
				normal_pval_p,
				levene_stat,
				levene_pval,
				ttest_stat,
				ttest_pval,
				mw_stat,
				mw_pval,
				cds)
allstatsr = zip(metric_names,
				means_h,
				means_pr,
				sd_h,
				sd_pr,
				normal_stat_h,
				normal_pval_h,
				normal_stat_pr,
				normal_pval_pr,
				levene_statr,
				levene_pvalr,
				ttest_statr,
				ttest_pvalr,
				mw_statr,
				mw_pvalr,
				cdsr)

df = pd.DataFrame(columns=["Metric",
			"Mean Healthy",
			"SD Healthy",
			"Mean Pred. Opt. Dose",
			"SD Pred. Opt. Dose",
			"ind. ttest stat",
			"ind. ttest p-value",
			"Cohen's d",
			"mwu stat",
			"mwu p-value",
			"levene stat",
			"leven p-value",
			"normal stat healthy",
			"normal p-value healthy",
			"normal stat pred. opt. dose",
			"normal p-value pred. opt. dose"])

for name,mh,mp,sdh,sdp,nsh,nph,nsp,npp,ls,lp,tts,ttp,mws,mwp,cd in allstats:
	df = df.append({"Metric":name,
			"Mean Healthy":mh,
			"SD Healthy":sdh,
			"Mean Pred. Opt. Dose":mp,
			"SD Pred. Opt. Dose":sdp,
			"ind. ttest stat":tts,
			"ind. ttest p-value":ttp,
			"Cohen's d":cd,
			"mwu stat":mws,
			"mwu p-value":mwp,
			"levene stat":ls,
			"leven p-value":lp,
			"normal stat healthy":nsh,
			"normal p-value healthy":nph,
			"normal stat pred. opt. dose":nsp,
			"normal p-value pred. opt. dose":npp},
			ignore_index = True)

df.to_csv('figs_ManualLinearRegression_PSDvalidation_V5_ANN/stats_predicted_dose_vs_healthy.csv')

df = pd.DataFrame(columns=["Metric",
			"Mean Healthy",
			"SD Healthy",
			"Mean Pred. Opt. Dose",
			"SD Pred. Opt. Dose",
			"ind. ttest stat",
			"ind. ttest p-value",
			"Cohen's d",
			"mwu stat",
			"mwu p-value",
			"levene stat",
			"leven p-value",
			"normal stat healthy",
			"normal p-value healthy",
			"normal stat pred. opt. dose",
			"normal p-value pred. opt. dose"])

for name,mh,mp,sdh,sdp,nsh,nph,nsp,npp,ls,lp,tts,ttp,mws,mwp,cd in allstatsr:
	df = df.append({"Metric":name,
			"Mean Healthy":mh,
			"SD Healthy":sdh,
			"Mean Pred. Opt. Dose":mp,
			"SD Pred. Opt. Dose":sdp,
			"ind. ttest stat":tts,
			"ind. ttest p-value":ttp,
			"Cohen's d":cd,
			"mwu stat":mws,
			"mwu p-value":mwp,
			"levene stat":ls,
			"leven p-value":lp,
			"normal stat healthy":nsh,
			"normal p-value healthy":nph,
			"normal stat pred. opt. dose":nsp,
			"normal p-value pred. opt. dose":npp},
			ignore_index = True)

df.to_csv('figs_ManualLinearRegression_PSDvalidation_V5_ANN/stats_rate_predicted_dose_vs_healthy.csv')

def plot_validation(metric_10sst,metric_20sst,metric_30sst,metric_40sst,healthy_range,ylabel,filename):
	fig_dosefits, ax_dosefits = plt.subplots(1, 1, sharey=True, figsize=(4,5))
	ax_dosefits.fill_between([0.7,2.3],y1=[healthy_range[1],healthy_range[1]],y2=[healthy_range[0],healthy_range[0]],color='k', alpha=0.2, zorder=2)
	count_recovered = 0
	for m1 in metric_10sst:
		ax_dosefits.plot([1,2],m1,color='royalblue',linewidth=2)
		if healthy_range[0] < m1[1] < healthy_range[1]: count_recovered+=1
	for m2 in metric_20sst:
		ax_dosefits.plot([1,2],m2,color='darkviolet',linewidth=2)
		if healthy_range[0] < m2[1] < healthy_range[1]: count_recovered+=1
	for m3 in metric_30sst:
		ax_dosefits.plot([1,2],m3,color='violet',linewidth=2)
		if healthy_range[0] < m3[1] < healthy_range[1]: count_recovered+=1
	for m4 in metric_40sst:
		ax_dosefits.plot([1,2],m4,color='deeppink',linewidth=2)
		if healthy_range[0] < m4[1] < healthy_range[1]: count_recovered+=1
	percent_recovered = (count_recovered/(len(metric_10sst)+len(metric_20sst)+len(metric_30sst)+len(metric_40sst)))*100
	y_loc = np.mean(np.transpose(metric_40sst)[0])
	ax_dosefits.text(2,y_loc,f"{percent_recovered:0.0f}"+'%',ha='center')
	ax_dosefits.set_xlim(0.7,2.3)
	ax_dosefits.set_xticks([1,2])
	ax_dosefits.set_xticklabels(labels=['MDD','+Dose'])
	ax_dosefits.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	ax_dosefits.set_ylabel(ylabel)
	fig_dosefits.tight_layout()
	fig_dosefits.savefig('figs_ManualLinearRegression_PSDvalidation_V5_ANN/'+filename+'.png',dpi=300,transparent=True)
	plt.close(fig_dosefits)

plot_validation(rate_mv_10sst,rate_mv_20sst,rate_mv_30sst,rate_mv_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_MVPredicted')
plot_validation(rate_aperiodic_10sst,rate_aperiodic_20sst,rate_aperiodic_30sst,rate_aperiodic_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_AperiodicPredicted')
plot_validation(rate_theta_10sst,rate_theta_20sst,rate_theta_30sst,rate_theta_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_ThetaPredicted')
plot_validation(rate_alpha_10sst,rate_alpha_20sst,rate_alpha_30sst,rate_alpha_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_AlphaPredicted')
plot_validation(rate_rate_10sst,rate_rate_20sst,rate_rate_30sst,rate_rate_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_RatePredicted')

plot_validation(failed_mv_10sst,failed_mv_20sst,failed_mv_30sst,failed_mv_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_MVPredicted')
plot_validation(failed_aperiodic_10sst,failed_aperiodic_20sst,failed_aperiodic_30sst,failed_aperiodic_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_AperiodicPredicted')
plot_validation(failed_theta_10sst,failed_theta_20sst,failed_theta_30sst,failed_theta_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_ThetaPredicted')
plot_validation(failed_alpha_10sst,failed_alpha_20sst,failed_alpha_30sst,failed_alpha_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_AlphaPredicted')
plot_validation(failed_rate_10sst,failed_rate_20sst,failed_rate_30sst,failed_rate_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_RatePredicted')

plot_validation(false_mv_10sst,false_mv_20sst,false_mv_30sst,false_mv_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_MVPredicted')
plot_validation(false_aperiodic_10sst,false_aperiodic_20sst,false_aperiodic_30sst,false_aperiodic_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_AperiodicPredicted')
plot_validation(false_theta_10sst,false_theta_20sst,false_theta_30sst,false_theta_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_ThetaPredicted')
plot_validation(false_alpha_10sst,false_alpha_20sst,false_alpha_30sst,false_alpha_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_AlphaPredicted')
plot_validation(false_rate_10sst,false_rate_20sst,false_rate_30sst,false_rate_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_RatePredicted')


def plot_validation_mean_CI(metric_10sst,metric_20sst,metric_30sst,metric_40sst,healthy_range,ylabel,filename):
	fig_dosefits, ax_dosefits = plt.subplots(1, 1, sharey=True, figsize=(4,5))
	ax_dosefits.fill_between([0.7,2.3],y1=[healthy_range[1],healthy_range[1]],y2=[healthy_range[0],healthy_range[0]],color='k', alpha=0.2, zorder=2)
	count_recovered = 0
	combined = metric_10sst + metric_20sst + metric_30sst + metric_40sst
	bootCI = False
	if bootCI:
		x1 = bs.bootstrap(np.transpose(combined)[0], stat_func=bs_stats.mean, alpha=0.1, num_iterations=50)
		combined_m1 = x1.value
		combined_l1 = combined_m1-x1.lower_bound
		combined_u1 = x1.upper_bound-combined_m1
		x2 = bs.bootstrap(np.transpose(combined)[1], stat_func=bs_stats.mean, alpha=0.1, num_iterations=50)
		combined_m2 = x2.value
		combined_l2 = combined_m2-x2.lower_bound
		combined_u2 = x2.upper_bound-combined_m2
		combined_m = [combined_m1,combined_m2]
		combined_l = [combined_l1,combined_l2]
		combined_u = [combined_u1,combined_u2]
		combined_err = (combined_l,combined_u)
	else:
		combined_m = np.mean(combined,axis=0)
		combined_err = np.std(combined,axis=0)
	
	for m1 in metric_10sst:
		ax_dosefits.plot([1,2],m1,color='royalblue',linewidth=2,alpha=0.1)
		if healthy_range[0] < m1[1] < healthy_range[1]: count_recovered+=1
	for m2 in metric_20sst:
		ax_dosefits.plot([1,2],m2,color='royalblue',linewidth=2,alpha=0.1)
		if healthy_range[0] < m2[1] < healthy_range[1]: count_recovered+=1
	for m3 in metric_30sst:
		ax_dosefits.plot([1,2],m3,color='darkviolet',linewidth=2,alpha=0.1)
		if healthy_range[0] < m3[1] < healthy_range[1]: count_recovered+=1
	for m4 in metric_40sst:
		ax_dosefits.plot([1,2],m4,color='deeppink',linewidth=2,alpha=0.1)
		if healthy_range[0] < m4[1] < healthy_range[1]: count_recovered+=1
	ax_dosefits.scatter([1,2],combined_m,s=10**2,color='k')
	ax_dosefits.errorbar([1,2],combined_m,yerr=combined_err,color='k',capsize=6,capthick=3,elinewidth=3, linestyle='')
	percent_recovered = (count_recovered/(len(metric_10sst)+len(metric_20sst)+len(metric_30sst)+len(metric_40sst)))*100
	y_loc = np.mean(np.transpose(metric_40sst)[0])
	ax_dosefits.text(2,y_loc,f"{percent_recovered:0.0f}"+'%',ha='center')
	ax_dosefits.set_xlim(0.7,2.3)
	ax_dosefits.set_xticks([1,2])
	ax_dosefits.set_xticklabels(labels=['MDD','+Dose'])
	ax_dosefits.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	ax_dosefits.set_ylabel(ylabel)
	fig_dosefits.tight_layout()
	fig_dosefits.savefig('figs_ManualLinearRegression_PSDvalidation_V5_ANN/mean_ci_'+filename+'.png',dpi=300,transparent=True)
	plt.close(fig_dosefits)

plot_validation_mean_CI(rate_mv_10sst,rate_mv_20sst,rate_mv_30sst,rate_mv_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_MVPredicted')
plot_validation_mean_CI(rate_aperiodic_10sst,rate_aperiodic_20sst,rate_aperiodic_30sst,rate_aperiodic_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_AperiodicPredicted')
plot_validation_mean_CI(rate_theta_10sst,rate_theta_20sst,rate_theta_30sst,rate_theta_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_ThetaPredicted')
plot_validation_mean_CI(rate_alpha_10sst,rate_alpha_20sst,rate_alpha_30sst,rate_alpha_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_AlphaPredicted')
plot_validation_mean_CI(rate_rate_10sst,rate_rate_20sst,rate_rate_30sst,rate_rate_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_RatePredicted')

plot_validation_mean_CI(failed_mv_10sst,failed_mv_20sst,failed_mv_30sst,failed_mv_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_MVPredicted')
plot_validation_mean_CI(failed_aperiodic_10sst,failed_aperiodic_20sst,failed_aperiodic_30sst,failed_aperiodic_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_AperiodicPredicted')
plot_validation_mean_CI(failed_theta_10sst,failed_theta_20sst,failed_theta_30sst,failed_theta_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_ThetaPredicted')
plot_validation_mean_CI(failed_alpha_10sst,failed_alpha_20sst,failed_alpha_30sst,failed_alpha_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_AlphaPredicted')
plot_validation_mean_CI(failed_rate_10sst,failed_rate_20sst,failed_rate_30sst,failed_rate_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_RatePredicted')

plot_validation_mean_CI(false_mv_10sst,false_mv_20sst,false_mv_30sst,false_mv_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_MVPredicted')
plot_validation_mean_CI(false_aperiodic_10sst,false_aperiodic_20sst,false_aperiodic_30sst,false_aperiodic_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_AperiodicPredicted')
plot_validation_mean_CI(false_theta_10sst,false_theta_20sst,false_theta_30sst,false_theta_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_ThetaPredicted')
plot_validation_mean_CI(false_alpha_10sst,false_alpha_20sst,false_alpha_30sst,false_alpha_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_AlphaPredicted')
plot_validation_mean_CI(false_rate_10sst,false_rate_20sst,false_rate_30sst,false_rate_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_RatePredicted')

def plot_validation_mean_CI_only(metric_10sst,metric_20sst,metric_30sst,metric_40sst,healthy_range,ylabel,filename):
	fig_dosefits, ax_dosefits = plt.subplots(1, 1, sharey=True, figsize=(4,5))
	ax_dosefits.fill_between([0.7,2.3],y1=[healthy_range[1],healthy_range[1]],y2=[healthy_range[0],healthy_range[0]],color='k', alpha=0.2, zorder=2)
	combined = metric_10sst + metric_20sst + metric_30sst + metric_40sst
	bootCI = False
	if bootCI:
		x1 = bs.bootstrap(np.transpose(combined)[0], stat_func=bs_stats.mean, alpha=0.1, num_iterations=50)
		combined_m1 = x1.value
		combined_l1 = combined_m1-x1.lower_bound
		combined_u1 = x1.upper_bound-combined_m1
		x2 = bs.bootstrap(np.transpose(combined)[1], stat_func=bs_stats.mean, alpha=0.1, num_iterations=50)
		combined_m2 = x2.value
		combined_l2 = combined_m2-x2.lower_bound
		combined_u2 = x2.upper_bound-combined_m2
		combined_m = [combined_m1,combined_m2]
		combined_l = [combined_l1,combined_l2]
		combined_u = [combined_u1,combined_u2]
		combined_err = (combined_l,combined_u)
	else:
		combined_m = np.mean(combined,axis=0)
		combined_err = np.std(combined,axis=0)
	ax_dosefits.scatter([1,2],combined_m,s=10**2,color='k')
	ax_dosefits.errorbar([1,2],combined_m,yerr=combined_err,color='k',capsize=6,capthick=3,elinewidth=3, linestyle='')
	ax_dosefits.set_xlim(0.7,2.3)
	ax_dosefits.set_xticks([1,2])
	ax_dosefits.set_xticklabels(labels=['MDD','+Dose'])
	ax_dosefits.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	ax_dosefits.set_ylabel(ylabel)
	fig_dosefits.tight_layout()
	fig_dosefits.savefig('figs_ManualLinearRegression_PSDvalidation_V5_ANN/mean_ci_only_'+filename+'.png',dpi=300,transparent=True)
	plt.close(fig_dosefits)

plot_validation_mean_CI_only(rate_mv_10sst,rate_mv_20sst,rate_mv_30sst,rate_mv_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_MVPredicted')
plot_validation_mean_CI_only(rate_aperiodic_10sst,rate_aperiodic_20sst,rate_aperiodic_30sst,rate_aperiodic_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_AperiodicPredicted')
plot_validation_mean_CI_only(rate_theta_10sst,rate_theta_20sst,rate_theta_30sst,rate_theta_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_ThetaPredicted')
plot_validation_mean_CI_only(rate_alpha_10sst,rate_alpha_20sst,rate_alpha_30sst,rate_alpha_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_AlphaPredicted')
plot_validation_mean_CI_only(rate_rate_10sst,rate_rate_20sst,rate_rate_30sst,rate_rate_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_RatePredicted')

plot_validation_mean_CI_only(failed_mv_10sst,failed_mv_20sst,failed_mv_30sst,failed_mv_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_MVPredicted')
plot_validation_mean_CI_only(failed_aperiodic_10sst,failed_aperiodic_20sst,failed_aperiodic_30sst,failed_aperiodic_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_AperiodicPredicted')
plot_validation_mean_CI_only(failed_theta_10sst,failed_theta_20sst,failed_theta_30sst,failed_theta_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_ThetaPredicted')
plot_validation_mean_CI_only(failed_alpha_10sst,failed_alpha_20sst,failed_alpha_30sst,failed_alpha_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_AlphaPredicted')
plot_validation_mean_CI_only(failed_rate_10sst,failed_rate_20sst,failed_rate_30sst,failed_rate_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_RatePredicted')

plot_validation_mean_CI_only(false_mv_10sst,false_mv_20sst,false_mv_30sst,false_mv_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_MVPredicted')
plot_validation_mean_CI_only(false_aperiodic_10sst,false_aperiodic_20sst,false_aperiodic_30sst,false_aperiodic_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_AperiodicPredicted')
plot_validation_mean_CI_only(false_theta_10sst,false_theta_20sst,false_theta_30sst,false_theta_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_ThetaPredicted')
plot_validation_mean_CI_only(false_alpha_10sst,false_alpha_20sst,false_alpha_30sst,false_alpha_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_AlphaPredicted')
plot_validation_mean_CI_only(false_rate_10sst,false_rate_20sst,false_rate_30sst,false_rate_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_RatePredicted')

def plot_validation_mean_CI_only_split(metric_10sst,metric_20sst,metric_30sst,metric_40sst,healthy_range,ylabel,filename,fs):
	fig_dosefits, ax_dosefits = plt.subplots(1, 1, sharey=True, figsize=fs)
	ax_dosefits.fill_between([0.7,2.3],y1=[healthy_range[1],healthy_range[1]],y2=[healthy_range[0],healthy_range[0]],color='k', alpha=0.2, zorder=2)
	combined = metric_10sst + metric_20sst + metric_30sst + metric_40sst
	
	combined_m = np.mean(combined,axis=0)
	metric_sst10_m = np.mean(metric_10sst,axis=0)
	metric_sst20_m = np.mean(metric_20sst,axis=0)
	metric_sst30_m = np.mean(metric_30sst,axis=0)
	metric_sst40_m = np.mean(metric_40sst,axis=0)
	combined_err = np.std(combined,axis=0)
	metric_sst10_err = np.std(metric_10sst,axis=0)
	metric_sst20_err = np.std(metric_20sst,axis=0)
	metric_sst30_err = np.std(metric_30sst,axis=0)
	metric_sst40_err = np.std(metric_40sst,axis=0)
	
	ax_dosefits.scatter([0.74,1.74],combined_m,s=10**2,color='k')
	ax_dosefits.errorbar([0.74,1.74],combined_m,yerr=combined_err,color='k',capsize=6,capthick=3,elinewidth=3, linestyle='')
	ax_dosefits.scatter([0.87,1.87],metric_sst10_m,s=10**2,color='royalblue')
	ax_dosefits.errorbar([0.87,1.87],metric_sst10_m,yerr=metric_sst10_err,color='royalblue',capsize=6,capthick=3,elinewidth=3, linestyle='')
	ax_dosefits.scatter([1,2],metric_sst20_m,s=10**2,color='darkviolet')
	ax_dosefits.errorbar([1,2],metric_sst20_m,yerr=metric_sst20_err,color='darkviolet',capsize=6,capthick=3,elinewidth=3, linestyle='')
	ax_dosefits.scatter([1.13,2.13],metric_sst30_m,s=10**2,color='violet')
	ax_dosefits.errorbar([1.13,2.13],metric_sst30_m,yerr=metric_sst30_err,color='violet',capsize=6,capthick=3,elinewidth=3, linestyle='')
	ax_dosefits.scatter([1.26,2.26],metric_sst40_m,s=10**2,color='deeppink')
	ax_dosefits.errorbar([1.26,2.26],metric_sst40_m,yerr=metric_sst40_err,color='deeppink',capsize=6,capthick=3,elinewidth=3, linestyle='')
	ax_dosefits.set_xlim(0.7,2.3)
	ax_dosefits.set_xticks([1,2])
	ax_dosefits.set_xticklabels(labels=['MDD','+Dose'])
	ax_dosefits.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	ax_dosefits.set_ylabel(ylabel)
	fig_dosefits.tight_layout()
	fig_dosefits.savefig('figs_ManualLinearRegression_PSDvalidation_V5_ANN/mean_ci_only_split_'+filename+'.png',dpi=300,transparent=True)
	plt.close(fig_dosefits)

plot_validation_mean_CI_only_split(rate_mv_10sst,rate_mv_20sst,rate_mv_30sst,rate_mv_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_MVPredicted',(4,5))
plot_validation_mean_CI_only_split(rate_aperiodic_10sst,rate_aperiodic_20sst,rate_aperiodic_30sst,rate_aperiodic_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_AperiodicPredicted',(4,5))
plot_validation_mean_CI_only_split(rate_theta_10sst,rate_theta_20sst,rate_theta_30sst,rate_theta_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_ThetaPredicted',(4,5))
plot_validation_mean_CI_only_split(rate_alpha_10sst,rate_alpha_20sst,rate_alpha_30sst,rate_alpha_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_AlphaPredicted',(4,5))
plot_validation_mean_CI_only_split(rate_rate_10sst,rate_rate_20sst,rate_rate_30sst,rate_rate_40sst,base_rates_lu[0][0],'Spike Rate (Hz)','validation_rate_RatePredicted',(5,4))

plot_validation_mean_CI_only_split(failed_mv_10sst,failed_mv_20sst,failed_mv_30sst,failed_mv_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_MVPredicted',(4,5),)
plot_validation_mean_CI_only_split(failed_aperiodic_10sst,failed_aperiodic_20sst,failed_aperiodic_30sst,failed_aperiodic_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_AperiodicPredicted',(4,5))
plot_validation_mean_CI_only_split(failed_theta_10sst,failed_theta_20sst,failed_theta_30sst,failed_theta_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_ThetaPredicted',(4,5))
plot_validation_mean_CI_only_split(failed_alpha_10sst,failed_alpha_20sst,failed_alpha_30sst,failed_alpha_40sst,failed_detections_lu[0][0],'Failed Detections (%)','validation_failed_AlphaPredicted',(4,5))
plot_validation_mean_CI_only_split(failed_rate_10sst,failed_rate_20sst,failed_rate_30sst,failed_rate_40sst,failed_detections_lu[0][0],'Failed (%)','validation_failed_RatePredicted',(5,4))

plot_validation_mean_CI_only_split(false_mv_10sst,false_mv_20sst,false_mv_30sst,false_mv_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_MVPredicted',(4,5))
plot_validation_mean_CI_only_split(false_aperiodic_10sst,false_aperiodic_20sst,false_aperiodic_30sst,false_aperiodic_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_AperiodicPredicted',(4,5))
plot_validation_mean_CI_only_split(false_theta_10sst,false_theta_20sst,false_theta_30sst,false_theta_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_ThetaPredicted',(4,5))
plot_validation_mean_CI_only_split(false_alpha_10sst,false_alpha_20sst,false_alpha_30sst,false_alpha_40sst,false_detections_lu[0][0],'False Detections (%)','validation_false_AlphaPredicted',(4,5))
plot_validation_mean_CI_only_split(false_rate_10sst,false_rate_20sst,false_rate_30sst,false_rate_40sst,false_detections_lu[0][0],'False (%)','validation_false_RatePredicted',(5,4))
