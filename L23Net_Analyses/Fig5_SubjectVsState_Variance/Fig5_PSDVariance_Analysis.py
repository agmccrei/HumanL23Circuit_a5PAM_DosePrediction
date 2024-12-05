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
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':26})

N_circuit_seeds = 10
N_state_seeds = 10
N_circuitseedsList = np.linspace(1,N_circuit_seeds,N_circuit_seeds, dtype=int)
N_stateseedsList = np.linspace(1,N_state_seeds,N_state_seeds, dtype=int)
N_cells = 1000
rate_threshold = 0.2 # Hz
dt = 0.025 # ms
startslice = 1000 # ms
endslice = 25000 # ms
t1 = int(startslice*(1/dt))
t2 = int(endslice*(1/dt))
tvec = np.arange(endslice/dt+1)*dt
nperseg = 80000 # len(tvec[t1:t2])/2

x_labels_types = ['Pyr', 'SST', 'PV', 'VIP']
colors_neurs = ['dimgrey', 'red', 'green', 'orange']

def cohen_d(y,x):
	nx = len(x)
	ny = len(y)
	dof = nx + ny - 2
	return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def scalebar(axis,xy,lw=3):
	# xy = [left,right,bottom,top]
	xscalebar = np.array([xy[0],xy[1],xy[1]])
	yscalebar = np.array([xy[2],xy[2],xy[3]])
	axis.plot(xscalebar,yscalebar,'k',linewidth=lw)

offsets = [[] for _ in N_circuitseedsList]
exponents = [[] for _ in N_circuitseedsList]
knees = [[] for _ in N_circuitseedsList]
errors = [[] for _ in N_circuitseedsList]
AUC = [[] for _ in N_circuitseedsList]

Ls = [[] for _ in N_circuitseedsList]
Gns = [[] for _ in N_circuitseedsList]

center_freqs_max = [[] for _ in N_circuitseedsList]
power_amps_max = [[] for _ in N_circuitseedsList]
bandwidths_max = [[] for _ in N_circuitseedsList]
AUC_max = [[] for _ in N_circuitseedsList]

center_freqs_t = [[] for _ in N_circuitseedsList]
power_amps_t = [[] for _ in N_circuitseedsList]
bandwidths_t = [[] for _ in N_circuitseedsList]
AUC_t = [[] for _ in N_circuitseedsList]
AUC_t_abs = [[] for _ in N_circuitseedsList]

center_freqs_a = [[] for _ in N_circuitseedsList]
power_amps_a = [[] for _ in N_circuitseedsList]
bandwidths_a = [[] for _ in N_circuitseedsList]
AUC_a = [[] for _ in N_circuitseedsList]
AUC_a_abs = [[] for _ in N_circuitseedsList]

center_freqs_b = [[] for _ in N_circuitseedsList]
power_amps_b = [[] for _ in N_circuitseedsList]
bandwidths_b = [[] for _ in N_circuitseedsList]
AUC_b = [[] for _ in N_circuitseedsList]
AUC_b_abs = [[] for _ in N_circuitseedsList]

AUC_broad_abs = [[] for _ in N_circuitseedsList]

thetaband = (4,8)
alphaband = (8,12)
betaband = (12,21)
broadband_ab = (8,21)
broadband = (3,30)

for seed_circuit in N_circuitseedsList:
	for seed_state in N_stateseedsList:
		print('Analyzing circuit seed #'+str(seed_circuit)+'234 and state seed #'+str(seed_state))
		
		temp_e = np.load('Saved_PSDs_AllSeeds/CircuitSeed' + str(seed_circuit) + '234StimSeed' + str(seed_state) + '_EEG.npy',allow_pickle=True)
		
		
		f_res = 1/(nperseg*dt/1000)
		freqrawEEG = np.arange(0,250,f_res)
		
		frange_init = 3
		frange = [frange_init,30]
		fm = FOOOF(peak_width_limits=(2, 6.),
					min_peak_height=0,
					aperiodic_mode='fixed',
					max_n_peaks=3,
					peak_threshold=2.)
		
		fm.fit(freqrawEEG, temp_e, frange)
		
		offsets[seed_circuit-1].append(fm.aperiodic_params_[0])
		exponents[seed_circuit-1].append(fm.aperiodic_params_[-1])
		errors[seed_circuit-1].append(fm.error_)
		
		L = 10**fm._ap_fit[np.where(fm.freqs>=frange_init)[0].tolist()[0]:]
		Gn = fm._peak_fit[np.where(fm.freqs>=frange_init)[0].tolist()[0]:]
		F = fm.freqs[np.where(fm.freqs>=frange_init)[0].tolist()[0]:]
		
		Ls[seed_circuit-1].append(L)
		Gns[seed_circuit-1].append(Gn)
		
		inds_ = [iids[0] for iids in np.argwhere((F>=broadband[0]) & (F<=broadband[1]))]
		inds_t = [iids[0] for iids in np.argwhere((F>=thetaband[0]) & (F<=thetaband[1]))]
		inds_a = [iids[0] for iids in np.argwhere((F>=alphaband[0]) & (F<=alphaband[1]))]
		inds_b = [iids[0] for iids in np.argwhere((F>=betaband[0]) & (F<=betaband[1]))]
		
		AUC[seed_circuit-1].append(np.trapz(L[inds_],x=F[inds_]))
		AUC_t[seed_circuit-1].append(np.trapz(Gn[inds_t],x=F[inds_t]))
		AUC_a[seed_circuit-1].append(np.trapz(Gn[inds_a],x=F[inds_a]))
		AUC_b[seed_circuit-1].append(np.trapz(Gn[inds_b],x=F[inds_b]))
		
		inds_ = [iids[0] for iids in np.argwhere((freqrawEEG>=broadband_ab[0]) & (freqrawEEG<=broadband_ab[1]))]
		inds_t = [iids[0] for iids in np.argwhere((freqrawEEG>=thetaband[0]) & (freqrawEEG<=thetaband[1]))]
		inds_a = [iids[0] for iids in np.argwhere((freqrawEEG>=alphaband[0]) & (freqrawEEG<=alphaband[1]))]
		inds_b = [iids[0] for iids in np.argwhere((freqrawEEG>=betaband[0]) & (freqrawEEG<=betaband[1]))]
		
		AUC_broad_abs[seed_circuit-1].append(np.trapz(temp_e[inds_],x=freqrawEEG[inds_]))
		AUC_t_abs[seed_circuit-1].append(np.trapz(temp_e[inds_t],x=freqrawEEG[inds_t]))
		AUC_a_abs[seed_circuit-1].append(np.trapz(temp_e[inds_a],x=freqrawEEG[inds_a]))
		AUC_b_abs[seed_circuit-1].append(np.trapz(temp_e[inds_b],x=freqrawEEG[inds_b]))
		
		prev_max = 0
		for ii in fm.peak_params_:
			cf = ii[0]
			pw = ii[1]
			bw = ii[2]
			if thetaband[0] <= cf <= thetaband[1]:
				center_freqs_t[seed_circuit-1].append(cf)
				power_amps_t[seed_circuit-1].append(pw)
				bandwidths_t[seed_circuit-1].append(bw)
			if alphaband[0] <= cf <= alphaband[1]:
				center_freqs_a[seed_circuit-1].append(cf)
				power_amps_a[seed_circuit-1].append(pw)
				bandwidths_a[seed_circuit-1].append(bw)
			if betaband[0] <= cf <= betaband[1]:
				center_freqs_b[seed_circuit-1].append(cf)
				power_amps_b[seed_circuit-1].append(pw)
				bandwidths_b[seed_circuit-1].append(bw)
			if pw > prev_max: # find max peak amplitude
				cf_max = cf # Max peak center frequency
				pw_max = pw # Max peak amplitude
				bw_max = bw # Max peak bandwidth
				inds_max = [iids[0] for iids in np.argwhere((F>=cf-bw/2) & (F<=cf+bw/2))]
				aw_max = np.trapz(Gn[inds_max],x=F[inds_max]) # Max peak AUC (using center frequency and bandwidth)
				prev_max = pw
		center_freqs_max[seed_circuit-1].append(cf_max)
		power_amps_max[seed_circuit-1].append(pw_max)
		bandwidths_max[seed_circuit-1].append(bw_max)
		AUC_max[seed_circuit-1].append(aw_max)

# Z-score and normalize data in log10 space
def normalize_results(input_data):
	data1 = np.log10(input_data*np.sign(input_data)) # log10
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

# Mean & SD state effects per circuit seed
def metric_variance_analysis(metric,label,filename):
	metric_circuit = metric
	metric_circuit_m = np.mean([element for sublist in metric_circuit for element in sublist]) # take mean of flattened list
	metric_circuit_sd = np.std([element for sublist in metric_circuit for element in sublist]) # take sd of flattened list
	
	metric_circuit_residuals = [[a-np.mean(allvals) for a in allvals] for allvals in metric]
	metric_circuit_residuals_m = np.mean([element for sublist in metric_circuit_residuals for element in sublist]) # take mean of flattened list
	metric_circuit_residuals_sd = np.std([element for sublist in metric_circuit_residuals for element in sublist]) # take sd of flattened list
	
	cmap = plt.cm.get_cmap('jet', len(metric_circuit_residuals))
	
	fig, ax = plt.subplots(figsize=(5.5,5))
	for c,(_,x,y) in enumerate(sorted(zip([np.mean(allvals) for allvals in metric],metric_circuit_residuals,metric_circuit))):
		ax.scatter(x,y,s=8**2,c=np.array([cmap(c)]))
	ax.scatter(metric_circuit_residuals_m,metric_circuit_m,s=8**2,c='k')
	ax.errorbar(metric_circuit_residuals_m,metric_circuit_m,xerr=metric_circuit_residuals_sd,yerr=metric_circuit_sd,ecolor='k', elinewidth=2, capthick=2, capsize=6, linestyle='')
	yscale = ax.get_ylim()
	ax.set_xlim(yscale - yscale[0] - (yscale[1]-yscale[0])/2)
	ax.set_xlabel('Centered ' + label + '\nAcross States')
	ax.set_ylabel('Circuit ' + label)
	#ax.grid()
	fig.tight_layout()
	fig.savefig('figs_EEG_V1/scatters_'+filename+'.png',dpi=300,transparent=True)
	plt.close()
	
	# Mean & SD circuit effects per state seed
	metric_state_m = [np.mean(allvals) for allvals in np.transpose(metric)]
	metric_state_sd = [np.std(allvals) for allvals in np.transpose(metric)]
	
	metric_circuit_m = [np.mean(allvals) for allvals in metric]
	metric_circuit_sd = [np.std(allvals) for allvals in metric]
	
	metric_state_variance_m = np.mean(metric_state_sd)
	metric_circuit_variance_m = np.mean(metric_circuit_sd)
	metric_state_variance_sd = np.std(metric_state_sd)
	metric_circuit_variance_sd = np.std(metric_circuit_sd)
	
	# Plot variance across states vs variance across circuits
	font = {'family' : 'normal',
			'weight' : 'normal',
			'size'   : 20}

	matplotlib.rc('font', **font)
	matplotlib.rc('legend',**{'fontsize':20})
	
	fig, ax = plt.subplots(figsize=(3.4,4.25))
	xvals = [0,1]
	ax.bar(xvals,[metric_state_variance_m,metric_circuit_variance_m],yerr=[metric_state_variance_sd,metric_circuit_variance_sd],width=0.7,color='grey',edgecolor='k',linewidth=3,error_kw=dict(lw=3, capsize=5, capthick=3))
	ax.set_xticks(xvals)
	ax.set_xticklabels(['Circuit','State'], rotation=45, ha='center')
	ax.set_ylabel('SD (Hz)')
	ax.set_xlim(-0.75,1.75)
	#ax.yaxis.tick_right()
	#ax.yaxis.set_label_position("right")
	fig.tight_layout()
	fig.savefig('figs_EEG_V1/scatters_Variance_'+filename+'.png',dpi=300,transparent=True)
	plt.close()
	
	g1_name = 'Circuit Variance'
	g1 = metric_state_sd
	g2_name = 'State Variance'
	g2 = metric_circuit_sd
	
	print(filename + ' stats:')
	print(g1_name+' mean = '+str(np.mean(g1)) + '\u00B1' + str(np.std(g1)))
	print(g2_name+' mean = '+str(np.mean(g2)) + '\u00B1' + str(np.std(g2)))
	print('Cohens D = ' + str(cohen_d(g1,g2)))
	if st.levene(g1,g2)[1] >= 0.05: print('Independent-Sample T-Test')
	else: print('Welsh T-Test')
	print('t-test p-value = ' + str(st.ttest_ind(g1,g2,equal_var=True if st.levene(g1,g2)[1] >= 0.05 else False)[1]))
	print('t-test t-stat = ' + str(st.ttest_ind(g1,g2,equal_var=True if st.levene(g1,g2)[1] >= 0.05 else False)[0]) + '\n')

metric_variance_analysis(offsets,'Offset','offsets')
metric_variance_analysis(exponents,r'$\chi$','exponents')
metric_variance_analysis(errors,'Errors','errors')
metric_variance_analysis(center_freqs_max,'Max Peak Frequency','centerfrequency_max')
metric_variance_analysis(power_amps_max,'Max Peak Amplitude','peakamplitude_max')
metric_variance_analysis(bandwidths_max,'Max Peak Broadband','bandwidth_max')
metric_variance_analysis(AUC_max,'Max Peak AUC','AUC_max')
metric_variance_analysis(AUC,'1/f','aperiodicAUC')
metric_variance_analysis(AUC_t,r'$\theta$','periodicthetaAUC')
metric_variance_analysis(AUC_a,r'$\alpha$','periodicalphaAUC')
metric_variance_analysis(AUC_b,r'$\beta$','periodicbetaAUC')
metric_variance_analysis(AUC_broad_abs,'Broadband','PSDbroadAUC')
metric_variance_analysis(AUC_t_abs,r'$\theta$','PSDthetaAUC')
metric_variance_analysis(AUC_a_abs,r'$\alpha$','PSDalphaAUC')
metric_variance_analysis(AUC_b_abs,r'$\beta$','PSDbetaAUC')
