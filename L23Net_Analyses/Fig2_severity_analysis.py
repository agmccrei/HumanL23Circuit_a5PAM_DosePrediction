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
doses = np.array([0.00])

conds = [['MDD_severity_'+f"{s:0.1f}"+'_a5PAM_'+f"{d:0.2f}"+'_long' for s in severities] for d in doses]
paths = [['Saved_PSDs_AllSeeds_nperseg_80000/'+i for i in c] for c in conds]

x_labels = [f"{s*100:0.0f}" + '%' for s in severities]
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

eeg = [[np.load(path + '_EEG.npy') for path in p] for p in paths]
spikes = [[np.load(path + '_spikes.npy') for path in p] for p in paths]
spikes_PN = [[np.load(path + '_spikes_PN.npy') for path in p] for p in paths]

offsets = [[[] for s in severities] for d in doses]
exponents = [[[] for s in severities] for d in doses]
knees = [[[] for s in severities] for d in doses]
errors = [[[] for s in severities] for d in doses]
AUC = [[[] for s in severities] for d in doses]

Ls = [[[] for s in severities] for d in doses]
Ls2 = [[[] for s in severities] for d in doses]
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
			
			L = 10**fm._ap_fit
			Gn = fm._peak_fit
			F = fm.freqs
			F2 = np.linspace(0.1,100,1000)
			L2 = 10**(fm.aperiodic_params_[0] - np.log10(F2**(fm.aperiodic_params_[-1])))
			
			Ls[cind1][cind2].append(L)
			Ls2[cind1][cind2].append(L2)
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

# Fit functions to dose-curves
def linear(x, slope, intercept):
	return slope*x + intercept

def expo(x, A, K, C):
	return A * np.exp(x * K) + C

def neg_expo(x, A, K, C):
	return A * np.exp(-x * K) + C

def sigmoid(x, k, L, b, x0):
	return (L / (1 + np.exp(k * (-x - x0)))) + b

def rev_sigmoid(x, k, L, b, x0):
	return (L / (1 + np.exp(k * (x - x0)))) + b

def run_line_fits(data,ylabel_plot,filename,direction='Negative'):
	xvals = severities
	xvals_highres = np.arange(xvals[0],xvals[-1],0.001)
	
	data_to_fit = [[],[]]
	for d,rc in zip(xvals,data):
		for r in rc:
			data_to_fit[0].append(d)
			data_to_fit[1].append(r)
	
	tstat,pval = st.pearsonr(data_to_fit[0],data_to_fit[1])
	
	p0_l = [tstat, np.max(data_to_fit[1])]
	p0_e = [1, 1, 1]
	p0_s = [2, 1, 1, 0.75]
	
	coeff_l = curve_fit(linear, data_to_fit[0], data_to_fit[1], p0 = p0_l, maxfev = 100000000, full_output=True)
	if direction == 'Negative':
		coeff_e = curve_fit(neg_expo, data_to_fit[0], data_to_fit[1], p0 = p0_e, maxfev = 100000000, full_output=True)
		coeff_s = curve_fit(rev_sigmoid, data_to_fit[0], data_to_fit[1], p0 = p0_s, maxfev = 200000000, full_output=True)
	elif direction == 'Positive':
		coeff_e = curve_fit(expo, data_to_fit[0], data_to_fit[1], p0 = p0_e, maxfev = 100000000, full_output=True)
		coeff_s = curve_fit(sigmoid, data_to_fit[0], data_to_fit[1], p0 = p0_s, maxfev = 200000000, full_output=True)
	
	SSE_l = np.around(sum((coeff_l[2]['fvec'])**2),3) if sum((coeff_l[2]['fvec'])**2) > 0.001 else np.format_float_scientific(sum((coeff_l[2]['fvec'])**2),4)
	SSE_e = np.around(sum((coeff_e[2]['fvec'])**2),3) if sum((coeff_e[2]['fvec'])**2) > 0.001 else np.format_float_scientific(sum((coeff_e[2]['fvec'])**2),4)
	SSE_s = np.around(sum((coeff_s[2]['fvec'])**2),3) if sum((coeff_s[2]['fvec'])**2) > 0.001 else np.format_float_scientific(sum((coeff_s[2]['fvec'])**2),4)
	
	l_fit = linear(xvals_highres, *coeff_l[0])
	if direction == 'Negative':
		e_fit = neg_expo(xvals_highres, *coeff_e[0])
		s_fit = rev_sigmoid(xvals_highres, *coeff_s[0])
	elif direction == 'Positive':
		e_fit = expo(xvals_highres, *coeff_e[0])
		s_fit = sigmoid(xvals_highres, *coeff_s[0])
	
	fig_dosefits, ax_dosefits = plt.subplots(figsize=(20, 8))
	ax_dosefits.scatter(data_to_fit[0], data_to_fit[1], c='k', label='Data')
	ax_dosefits.plot(xvals_highres, l_fit, c='r', lw=8, ls=':', label='Linear SSE: '+str(SSE_l) + '; R,p = ' + str(np.round(tstat,3)) + ',' + str(np.format_float_scientific(pval,3)))
	ax_dosefits.plot(xvals_highres, e_fit, c='b', lw=6, ls='--', label='Exponential SSE: '+str(SSE_e))
	ax_dosefits.plot(xvals_highres, s_fit, c='g', lw=4, ls='-.', label='Sigmoidal SSE: '+str(SSE_s))
	ax_dosefits.set_xlabel('Severity Mulitplier')
	ax_dosefits.set_ylabel(ylabel_plot)
	ax_dosefits.legend()
	fig_dosefits.tight_layout()
	fig_dosefits.savefig('figs_PSD_SeverityOnly_V2/'+filename+'.png',dpi=300,transparent=True)
	plt.close()

run_line_fits(exponents[0],r'$\chi$','line_fits_severity_exponents',direction='Negative')
run_line_fits(AUC[0],'Ap. Power','line_fits_severity_aperiodic_AUC',direction='Positive')
run_line_fits(AUC_t[0],r'$\theta$','line_fits_severity_periodic_theta',direction='Positive')
run_line_fits(AUC_a[0],r'$\alpha$','line_fits_severity_periodic_alpha',direction='Positive')
run_line_fits(AUC_b[0],r'$\beta$','line_fits_severity_periodic_beta',direction='Positive')
run_line_fits(AUC_t_abs[0],r'$\theta$ Power','line_fits_severity_EEG_theta',direction='Positive')
run_line_fits(AUC_a_abs[0],r'$\alpha$ Power','line_fits_severity_EEG_alpha',direction='Positive')
run_line_fits(AUC_b_abs[0],r'$\beta$ Power','line_fits_severity_EEG_beta',direction='Positive')

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

# vs healthy
offsets_tstat0 = [[st.ttest_rel(offsets[0][0],offsets[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
exponents_tstat0 = [[st.ttest_rel(exponents[0][0],exponents[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
errors_tstat0 = [[st.ttest_rel(errors[0][0],errors[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_t_tstat0 = [[st.ttest_ind(center_freqs_t[0][0],center_freqs_t[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_t_tstat0 = [[st.ttest_ind(power_amps_t[0][0],power_amps_t[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_t_tstat0 = [[st.ttest_ind(bandwidths_t[0][0],bandwidths_t[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_a_tstat0 = [[st.ttest_ind(center_freqs_a[0][0],center_freqs_a[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_a_tstat0 = [[st.ttest_ind(power_amps_a[0][0],power_amps_a[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_a_tstat0 = [[st.ttest_ind(bandwidths_a[0][0],bandwidths_a[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_b_tstat0 = [[st.ttest_ind(center_freqs_b[0][0],center_freqs_b[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_b_tstat0 = [[st.ttest_ind(power_amps_b[0][0],power_amps_b[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_b_tstat0 = [[st.ttest_ind(bandwidths_b[0][0],bandwidths_b[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_max_tstat0 = [[st.ttest_ind(center_freqs_max[0][0],center_freqs_max[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_max_tstat0 = [[st.ttest_ind(power_amps_max[0][0],power_amps_max[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_max_tstat0 = [[st.ttest_ind(bandwidths_max[0][0],bandwidths_max[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_max_tstat0 = [[st.ttest_ind(AUC_max[0][0],AUC_max[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_tstat0 = [[st.ttest_rel(AUC[0][0],AUC[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_t_tstat0 = [[st.ttest_rel(AUC_t[0][0],AUC_t[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_a_tstat0 = [[st.ttest_rel(AUC_a[0][0],AUC_a[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_b_tstat0 = [[st.ttest_rel(AUC_b[0][0],AUC_b[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_broad_abs_tstat0 = [[st.ttest_rel(AUC_broad_abs[0][0],AUC_broad_abs[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_t_abs_tstat0 = [[st.ttest_rel(AUC_t_abs[0][0],AUC_t_abs[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_a_abs_tstat0 = [[st.ttest_rel(AUC_a_abs[0][0],AUC_a_abs[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_b_abs_tstat0 = [[st.ttest_rel(AUC_b_abs[0][0],AUC_b_abs[d][s])[0] for s in range(0,len(severities))] for d in range(0,len(doses))]

offsets_p0 = [[st.ttest_rel(offsets[0][0],offsets[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
exponents_p0 = [[st.ttest_rel(exponents[0][0],exponents[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
errors_p0 = [[st.ttest_rel(errors[0][0],errors[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_t_p0 = [[st.ttest_ind(center_freqs_t[0][0],center_freqs_t[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_t_p0 = [[st.ttest_ind(power_amps_t[0][0],power_amps_t[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_t_p0 = [[st.ttest_ind(bandwidths_t[0][0],bandwidths_t[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_a_p0 = [[st.ttest_ind(center_freqs_a[0][0],center_freqs_a[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_a_p0 = [[st.ttest_ind(power_amps_a[0][0],power_amps_a[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_a_p0 = [[st.ttest_ind(bandwidths_a[0][0],bandwidths_a[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_b_p0 = [[st.ttest_ind(center_freqs_b[0][0],center_freqs_b[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_b_p0 = [[st.ttest_ind(power_amps_b[0][0],power_amps_b[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_b_p0 = [[st.ttest_ind(bandwidths_b[0][0],bandwidths_b[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_max_p0 = [[st.ttest_ind(center_freqs_max[0][0],center_freqs_max[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_max_p0 = [[st.ttest_ind(power_amps_max[0][0],power_amps_max[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_max_p0 = [[st.ttest_ind(bandwidths_max[0][0],bandwidths_max[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_max_p0 = [[st.ttest_ind(AUC_max[0][0],AUC_max[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_p0 = [[st.ttest_rel(AUC[0][0],AUC[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_t_p0 = [[st.ttest_rel(AUC_t[0][0],AUC_t[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_a_p0 = [[st.ttest_rel(AUC_a[0][0],AUC_a[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_b_p0 = [[st.ttest_rel(AUC_b[0][0],AUC_b[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_broad_abs_p0 = [[st.ttest_rel(AUC_broad_abs[0][0],AUC_broad_abs[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_t_abs_p0 = [[st.ttest_rel(AUC_t_abs[0][0],AUC_t_abs[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_a_abs_p0 = [[st.ttest_rel(AUC_a_abs[0][0],AUC_a_abs[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_b_abs_p0 = [[st.ttest_rel(AUC_b_abs[0][0],AUC_b_abs[d][s])[1] for s in range(0,len(severities))] for d in range(0,len(doses))]

offsets_cd0 = [[cohen_d(offsets[0][0],offsets[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
exponents_cd0 = [[cohen_d(exponents[0][0],exponents[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
errors_cd0 = [[cohen_d(errors[0][0],errors[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_t_cd0 = [[cohen_d(center_freqs_t[0][0],center_freqs_t[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_t_cd0 = [[cohen_d(power_amps_t[0][0],power_amps_t[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_t_cd0 = [[cohen_d(bandwidths_t[0][0],bandwidths_t[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_a_cd0 = [[cohen_d(center_freqs_a[0][0],center_freqs_a[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_a_cd0 = [[cohen_d(power_amps_a[0][0],power_amps_a[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_a_cd0 = [[cohen_d(bandwidths_a[0][0],bandwidths_a[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_b_cd0 = [[cohen_d(center_freqs_b[0][0],center_freqs_b[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_b_cd0 = [[cohen_d(power_amps_b[0][0],power_amps_b[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_b_cd0 = [[cohen_d(bandwidths_b[0][0],bandwidths_b[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
center_freqs_max_cd0 = [[cohen_d(center_freqs_max[0][0],center_freqs_max[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
power_amps_max_cd0 = [[cohen_d(power_amps_max[0][0],power_amps_max[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
bandwidths_max_cd0 = [[cohen_d(bandwidths_max[0][0],bandwidths_max[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_max_cd0 = [[cohen_d(AUC_max[0][0],AUC_max[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_cd0 = [[cohen_d(AUC[0][0],AUC[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_t_cd0 = [[cohen_d(AUC_t[0][0],AUC_t[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_a_cd0 = [[cohen_d(AUC_a[0][0],AUC_a[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_b_cd0 = [[cohen_d(AUC_b[0][0],AUC_b[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_broad_abs_cd0 = [[cohen_d(AUC_broad_abs[0][0],AUC_broad_abs[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_t_abs_cd0 = [[cohen_d(AUC_t_abs[0][0],AUC_t_abs[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_a_abs_cd0 = [[cohen_d(AUC_a_abs[0][0],AUC_a_abs[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]
AUC_b_abs_cd0 = [[cohen_d(AUC_b_abs[0][0],AUC_b_abs[d][s]) for s in range(0,len(severities))] for d in range(0,len(doses))]

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
				"Center Frequency (Theta)",
				"Relative Power (Theta)",
				"Bandwidth (Theta)",
				"Center Frequency (alpha)",
				"Relative Power (alpha)",
				"Bandwidth (alpha)",
				"Center Frequency (beta)",
				"Relative Power (beta)",
				"Bandwidth (beta)",
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
	center_freqs_t_m,
	power_amps_t_m,
	bandwidths_t_m,
	center_freqs_a_m,
	power_amps_a_m,
	bandwidths_a_m,
	center_freqs_b_m,
	power_amps_b_m,
	bandwidths_b_m,
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
	center_freqs_t_sd,
	power_amps_t_sd,
	bandwidths_t_sd,
	center_freqs_a_sd,
	power_amps_a_sd,
	bandwidths_a_sd,
	center_freqs_b_sd,
	power_amps_b_sd,
	bandwidths_b_sd,
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
	center_freqs_t_tstat0,
	power_amps_t_tstat0,
	bandwidths_t_tstat0,
	center_freqs_a_tstat0,
	power_amps_a_tstat0,
	bandwidths_a_tstat0,
	center_freqs_b_tstat0,
	power_amps_b_tstat0,
	bandwidths_b_tstat0,
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
	center_freqs_t_p0,
	power_amps_t_p0,
	bandwidths_t_p0,
	center_freqs_a_p0,
	power_amps_a_p0,
	bandwidths_a_p0,
	center_freqs_b_p0,
	power_amps_b_p0,
	bandwidths_b_p0,
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
	center_freqs_t_cd0,
	power_amps_t_cd0,
	bandwidths_t_cd0,
	center_freqs_a_cd0,
	power_amps_a_cd0,
	bandwidths_a_cd0,
	center_freqs_b_cd0,
	power_amps_b_cd0,
	bandwidths_b_cd0,
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

df.to_csv('figs_PSD_SeverityOnly_V2/stats_PSD.csv')

figsize1 = (7,5)
figsize2 = (9,5)
figsize3 = (6,6)
dh1 = 0.03

p_thresh = 0.05
c_thresh = 1.

cmap = plt.cm.get_cmap('hot')#.reversed()
norm = plt.Normalize(vmin=severities[0], vmax=severities[-1]+0.26)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

def plot_responsecurves_allseverities(data,data_m,data_sd,data_p0,data_cd0,ylabel_plot,filename,ylims=False,figsizes=figsize1):
	xinds = np.arange(0,len(severities))
	cind1 = 0
	
	fig_bands, ax_bands = plt.subplots(figsize=figsizes)
	
	ax_bands.plot([xinds[0]-0.5, xinds[-1]+0.5],[data_m[0][0],data_m[0][0]],color=sm.to_rgba(severities[0]),ls='dashed',alpha=1)
	ax_bands.fill_between([xinds[0]-0.5, xinds[-1]+0.5], y1 = [data_m[0][0]+data_sd[0][0],data_m[0][0]+data_sd[0][0]], y2 = [data_m[0][0]-data_sd[0][0],data_m[0][0]-data_sd[0][0]], color=sm.to_rgba(severities[0]), alpha=0.2, zorder=2)
	
	for cind2,s in enumerate(severities):
		ax_bands.scatter(xinds[cind2]+(np.random.random(len(data[cind1][cind2]))*0.4-0.2),data[cind1][cind2],s=23,facecolor=sm.to_rgba(severities[cind2]),edgecolors='face')
		ln1, = ax_bands.plot([xinds[cind2]-0.05,xinds[cind2]+0.05],[data_m[cind1][cind2],data_m[cind1][cind2]],'k',alpha=1,linewidth=3)
		ln1.set_solid_capstyle('round')
		ax_bands.errorbar(xinds[cind2],data_m[cind1][cind2],yerr=data_sd[cind1][cind2],color='k', fmt='', capsize=10, linewidth=3, capthick=3)

		if ((data_p0[cind1][cind2] < p_thresh) & (abs(data_cd0[cind1][cind2]) > c_thresh)):
			ax_bands.text(xinds[cind2],data_m[cind1][cind2]+data_sd[cind1][cind2]+data_sd[cind1][cind2],'*',c='k',ha='center', va='bottom',fontweight='bold')
	
	ax_bands.set_ylabel(ylabel_plot)
	if ylims is not False: ax_bands.set_ylim(ylims)
	if filename in ['EEG_PSDthetaAUC','EEG_PSDalphaAUC','EEG_PSDbetaAUC','EEG_aperiodicAUC']:
		ax_bands.yaxis.set_major_locator(mticker.MultipleLocator(0.25e-13))
		ax_bands.set_xticks(xinds)
		ax_bands.set_xticklabels([0,10,20,30,40], rotation = 0, ha="center")
	else:
		ax_bands.set_xticks(xinds)
		ax_bands.set_xticklabels(x_labels, rotation = 0, ha="center")
	ax_bands.grid(False)
	ax_bands.spines['right'].set_visible(False)
	ax_bands.spines['top'].set_visible(False)
	fig_bands.tight_layout()
	fig_bands.savefig('figs_PSD_SeverityOnly_V2/scatters_severity_'+filename+'.png',dpi=300,transparent=True)
	plt.close()

plot_responsecurves_allseverities(offsets,offsets_m,offsets_sd,offsets_p0,offsets_cd0,r'Offset (mV$^2$)','EEG_offsets',figsizes=figsize2)#,(-14,-13.1))
plot_responsecurves_allseverities(AUC_t_abs,AUC_t_abs_m,AUC_t_abs_sd,AUC_t_abs_p0,AUC_t_abs_cd0,r'$\theta$'+' Power','EEG_PSDthetaAUC',figsizes=figsize3)#,(1*10**-14,1.27*10**-13))
plot_responsecurves_allseverities(AUC_a_abs,AUC_a_abs_m,AUC_a_abs_sd,AUC_a_abs_p0,AUC_a_abs_cd0,r'$\alpha$'+' Power','EEG_PSDalphaAUC',figsizes=figsize3)#,(1.7*10**-14,1.8*10**-13))
plot_responsecurves_allseverities(AUC_b_abs,AUC_b_abs_m,AUC_b_abs_sd,AUC_b_abs_p0,AUC_b_abs_cd0,r'$\beta$'+' Power','EEG_PSDbetaAUC',figsizes=figsize3)#,(0.8*10**-14,1.2*10**-13))
plot_responsecurves_allseverities(AUC,AUC_m,AUC_sd,AUC_p0,AUC_cd0,'1/f Power','EEG_aperiodicAUC',figsizes=figsize3)#,(0.25*10**-13,1.75*10**-13))
plot_responsecurves_allseverities(AUC_t,AUC_t_m,AUC_t_sd,AUC_t_p0,AUC_t_cd0,r'$\theta$','EEG_thetaAUC',figsizes=figsize2)#,(0.5,2.2))
plot_responsecurves_allseverities(AUC_a,AUC_a_m,AUC_a_sd,AUC_a_p0,AUC_a_cd0,r'$\alpha$','EEG_alphaAUC',figsizes=figsize2)#,(1.0,3.1))
plot_responsecurves_allseverities(AUC_b,AUC_b_m,AUC_b_sd,AUC_b_p0,AUC_b_cd0,r'$\beta$','EEG_betaAUC',figsizes=figsize2)#,(0.0,3.0))
plot_responsecurves_allseverities(exponents,exponents_m,exponents_sd,exponents_p0,exponents_cd0,r'$\chi$','EEG_exponents',figsizes=figsize2)#,(0.4,1.2))
plot_responsecurves_allseverities(errors,errors_m,errors_sd,errors_p0,errors_cd0,'Error','EEG_errors',figsizes=figsize2)#,(0.077,0.14))
plot_responsecurves_allseverities(center_freqs_t,center_freqs_t_m,center_freqs_t_sd,center_freqs_t_p0,center_freqs_t_cd0,r'$\theta$'+' Center Frequency (Hz)','centerfrequency_theta',figsizes=figsize2)
plot_responsecurves_allseverities(center_freqs_a,center_freqs_a_m,center_freqs_a_sd,center_freqs_a_p0,center_freqs_a_cd0,r'$\alpha$'+' Center Frequency (Hz)','centerfrequency_alpha',figsizes=figsize2)
plot_responsecurves_allseverities(center_freqs_b,center_freqs_b_m,center_freqs_b_sd,center_freqs_b_p0,center_freqs_b_cd0,r'$\beta$'+' Center Frequency (Hz)','centerfrequency_beta',figsizes=figsize2)
plot_responsecurves_allseverities(power_amps_t,power_amps_t_m,power_amps_t_sd,power_amps_t_p0,power_amps_t_cd0,r'$\theta$'+' Peak Amplitude','peakamplitude_theta',figsizes=figsize2)
plot_responsecurves_allseverities(power_amps_a,power_amps_a_m,power_amps_a_sd,power_amps_a_p0,power_amps_a_cd0,r'$\alpha$'+' Peak Amplitude','peakamplitude_alpha',figsizes=figsize2)
plot_responsecurves_allseverities(power_amps_b,power_amps_b_m,power_amps_b_sd,power_amps_b_p0,power_amps_b_cd0,r'$\beta$'+' Peak Amplitude','peakamplitude_beta',figsizes=figsize2)
plot_responsecurves_allseverities(bandwidths_t,bandwidths_t_m,bandwidths_t_sd,bandwidths_t_p0,bandwidths_t_cd0,r'$\theta$'+' Bandwidth (Hz)','bandwidth_theta',figsizes=figsize2)
plot_responsecurves_allseverities(bandwidths_a,bandwidths_a_m,bandwidths_a_sd,bandwidths_a_p0,bandwidths_a_cd0,r'$\alpha$'+' Bandwidth (Hz)','bandwidth_alpha',figsizes=figsize2)
plot_responsecurves_allseverities(bandwidths_b,bandwidths_b_m,bandwidths_b_sd,bandwidths_b_p0,bandwidths_b_cd0,r'$\beta$'+' Bandwidth (Hz)','bandwidth_beta',figsizes=figsize2)
plot_responsecurves_allseverities(center_freqs_max,center_freqs_max_m,center_freqs_max_sd,center_freqs_max_p0,center_freqs_max_cd0,'Max Peak Frequency (Hz)','centerfrequency_max',figsizes=figsize2)
plot_responsecurves_allseverities(power_amps_max,power_amps_max_m,power_amps_max_sd,power_amps_max_p0,power_amps_max_cd0,'Max Peak Amplitude','peakamplitude_max',figsizes=figsize2)
plot_responsecurves_allseverities(bandwidths_max,bandwidths_max_m,bandwidths_max_sd,bandwidths_max_p0,bandwidths_max_cd0,'Max Peak Bandwidth (Hz)','bandwidth_max',figsizes=figsize2)
plot_responsecurves_allseverities(AUC_max,AUC_max_m,AUC_max_sd,AUC_max_p0,AUC_max_cd0,'Max Peak AUC','AUC_max',figsizes=figsize2)

		
freq0 = 3
freq1 = 30
freq2 = 100
f1 = np.where(freqrawEEG>=freq0)
f1 = f1[0][0]-1
f2 = np.where(freqrawEEG>=freq1)
f2 = f2[0][0]+1
f3 = np.where(freqrawEEG>=freq2)
f3 = f3[0][0]

# Plot EEG
bootCI = True
for cind1, d in enumerate(doses):
	fig_eeg, ax_eeg = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=True, figsize=(7, 5))
	inset = ax_eeg.inset_axes([.54,.45,.42,.5])
	# inset = ax_eeg.inset_axes([.5,.45,.45,.5])
	for cind2, s in enumerate(severities):
		if bootCI:
			CI_means_EEG = []
			CI_lower_EEG = []
			CI_upper_EEG = []
			CI_means_in = []
			CI_lower_in = []
			CI_upper_in = []
			ir = [freq0,freq1]
			F_in = freqrawEEG[(freqrawEEG>=ir[0])&(freqrawEEG<=ir[1])]
			for l in range(0,len(eeg[cind1][cind2][0])):
				x = bs.bootstrap(np.transpose(eeg[cind1][cind2])[l], stat_func=bs_stats.mean, alpha=0.1, num_iterations=50)
				CI_means_EEG.append(x.value)
				CI_lower_EEG.append(x.lower_bound)
				CI_upper_EEG.append(x.upper_bound)
				if freqrawEEG[l] < ir[0]:
					continue
				elif freqrawEEG[l] > ir[1]:
					continue
				else:
					CI_means_in.append(x.value)
					CI_lower_in.append(x.lower_bound)
					CI_upper_in.append(x.upper_bound)
		else:
			CI_means_EEG = np.mean(eeg[cind1][cind2],0)
			CI_lower_EEG = np.mean(eeg[cind1][cind2],0)-np.std(eeg[cind1][cind2],0)
			CI_upper_EEG = np.mean(eeg[cind1][cind2],0)+np.std(eeg[cind1][cind2],0)
		
		ax_eeg.plot(freqrawEEG, CI_means_EEG, color=sm.to_rgba(severities[cind2]),linewidth=3)
		ax_eeg.fill_between(freqrawEEG, CI_lower_EEG, CI_upper_EEG, color=sm.to_rgba(severities[cind2]),alpha=0.3)
		ax_eeg.tick_params(axis='x', which='major', bottom=True)
		ax_eeg.tick_params(axis='y', which='major', left=True)
		inset.plot(F_in, CI_means_in, color=sm.to_rgba(severities[cind2]),linewidth=3)
		inset.set_xscale('log')
		inset.set_yscale('log')
		inset.tick_params(axis='x', which='major', bottom=True)
		inset.tick_params(axis='y', which='major', left=True)
		inset.tick_params(axis='x', which='minor', bottom=True)
		inset.tick_params(axis='y', which='minor', left=True)
		inset.xaxis.set_minor_locator(LogLocator(base=10, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
		inset.yaxis.set_minor_locator(LogLocator(base=10, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
		inset.fill_between(F_in, CI_lower_in, CI_upper_in,color=sm.to_rgba(severities[cind2]),alpha=0.3)
		inset.set_xticks(ir)
		inset.set_xlim(ir[0],ir[1])
		#inset.set_xlim(0.3,freq2)
		inset.xaxis.set_major_formatter(mticker.ScalarFormatter())
		inset.xaxis.set_minor_formatter(mticker.NullFormatter())

	ylims = ax_eeg.get_ylim()
	ax_eeg.set_ylim(0,3.1*10**-14)
	ax_eeg.set_xticks([freq0,5,10,15,20,25,freq1])
	ax_eeg.set_xticklabels(['','5','10','15','20','25',str(freq1)])
	ax_eeg.set_xlim(freq0,freq1)
	ax_eeg.set_xlabel('Frequency (Hz)')
	ax_eeg.set_ylabel('Power (mV'+r'$^{2}$'+'/Hz)')
	ax_eeg.text(thetaband[0]+(np.diff(thetaband)/2)-0.75,0.1*10**-14,r'$\theta$',fontsize=24)
	ax_eeg.text(alphaband[0]+(np.diff(alphaband)/2)-0.75,0.1*10**-14,r'$\alpha$',fontsize=24)
	ax_eeg.text(13.25,0.1*10**-14,r'$\beta$',fontsize=24)
	ylims_2 = ax_eeg.get_ylim()
	ax_eeg.plot([thetaband[0],thetaband[0]],ylims_2,c='dimgrey',ls=':')
	ax_eeg.plot([alphaband[0],alphaband[0]],ylims_2,c='dimgrey',ls=':')
	ax_eeg.plot([alphaband[1],alphaband[1]],ylims_2,c='dimgrey',ls=':')
	ax_eeg.set_ylim(ylims_2)
	ax_eeg.set_xlim(ir[0],ir[1])
	
	fig_eeg.savefig('figs_PSD_SeverityOnly_V2/severity_EEG_PSD.png',bbox_inches='tight',dpi=300,transparent=True)
	plt.close()


# Plot aperiodic components
for cind1, d in enumerate(doses):
	fig_bands, ax_bands = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=True, figsize=(7, 5))
	inset = ax_bands.inset_axes([.54,.45,.42,.5])
	# inset = ax_bands.inset_axes([.5,.45,.45,.5])
	for cind2, s in enumerate(severities):
		CI_means = []
		CI_lower = []
		CI_upper = []
		CI_means_in = []
		CI_lower_in = []
		CI_upper_in = []
		#ir = [9,freq2]
		ir = [freq0,freq1]
		F_in = F2[(F2>=ir[0])&(F2<=ir[1])]
		for l in range(0,len(Ls2[cind1][cind2][0])):
			x = bs.bootstrap(np.transpose(Ls2[cind1][cind2])[l], stat_func=bs_stats.mean, alpha=0.1, num_iterations=50)
			CI_means.append(x.value)
			CI_lower.append(x.lower_bound)
			CI_upper.append(x.upper_bound)
			if F2[l] < ir[0]:
				continue
			elif F2[l] > ir[1]:
				continue
			else:
				CI_means_in.append(x.value)
				CI_lower_in.append(x.lower_bound)
				CI_upper_in.append(x.upper_bound)
		ax_bands.fill_between(F2,CI_lower,CI_upper, color=sm.to_rgba(severities[cind2]),alpha=0.3)
		ax_bands.plot(F2,CI_means,color=sm.to_rgba(severities[cind2]),linewidth=3)
		
		inset.plot(F_in,CI_means_in, color=sm.to_rgba(severities[cind2]),linewidth=3)
		inset.set_xscale('log')
		inset.set_yscale('log')
		inset.tick_params(axis='x', which='major', bottom=True)
		inset.tick_params(axis='y', which='major', left=True)
		inset.tick_params(axis='x', which='minor', bottom=True)
		inset.tick_params(axis='y', which='minor', left=True)
		inset.xaxis.set_minor_locator(LogLocator(base=10, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
		inset.yaxis.set_minor_locator(LogLocator(base=10, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
		inset.fill_between(F_in,CI_lower_in,CI_upper_in,color=sm.to_rgba(severities[cind2]),alpha=0.3)
		inset.set_xticks(ir)
		inset.set_xlim(ir[0],ir[1])
		inset.xaxis.set_major_formatter(mticker.ScalarFormatter())
		inset.xaxis.set_minor_formatter(mticker.NullFormatter())

	ax_bands.tick_params(axis='x', which='major', bottom=True)
	ax_bands.tick_params(axis='y', which='major', left=True)
	ax_bands.set_xticks([freq0,5,10,15,20,25,freq1])
	ax_bands.set_xticklabels(['','5','10','15','20','25',str(freq1)])
	ax_bands.set_xlabel('Frequency (Hz)')
	ax_bands.set_ylabel('Power (mV'+r'$^{2}$'+'/Hz)')
	
	ax_bands.set_ylim(0,2*10**-14)
	ylims = ax_bands.get_ylim()
	ax_bands.set_ylim(ylims)
	ax_bands.set_xlim(ir[0],ir[1])
	
	fig_bands.savefig('figs_PSD_SeverityOnly_V2/severity_aperiodic.png',bbox_inches='tight',dpi=300,transparent=True)
	plt.close()

# Plot periodic components
for cind1, d in enumerate(doses):
	fig_bands, ax_bands = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=True, figsize=(7, 5))
	for cind2, s in enumerate(severities):
		
		CI_means = []
		CI_lower = []
		CI_upper = []
		for l in range(0,len(Gns[cind1][cind2][0])):
			x = bs.bootstrap(np.transpose(Gns[cind1][cind2])[l], stat_func=bs_stats.mean, alpha=0.1, num_iterations=50)
			CI_means.append(x.value)
			CI_lower.append(x.lower_bound)
			CI_upper.append(x.upper_bound)
		ax_bands.fill_between(F,CI_lower,CI_upper, color=sm.to_rgba(severities[cind2]),alpha=0.3)
		ax_bands.plot(F,CI_means,color=sm.to_rgba(severities[cind2]),linewidth=3)
	
	ax_bands.tick_params(axis='x', which='major', bottom=True)
	ax_bands.tick_params(axis='y', which='major', left=True)
	ax_bands.set_xticks([freq0,5,10,15,20,25,freq1])
	ax_bands.set_xticklabels(['','5','10','15','20','25',str(freq1)])
	ax_bands.set_xlabel('Frequency (Hz)')
	ax_bands.set_ylabel(r'log(Power) - log(Power$_{AP}$)')
	
	ylims = ax_bands.get_ylim()
	ax_bands.set_ylim(0,0.7)
	ax_bands.text(thetaband[0]+(np.diff(thetaband)/2)-0.75,0.02,r'$\theta$',fontsize=24)
	ax_bands.text(alphaband[0]+(np.diff(alphaband)/2)-0.75,0.02,r'$\alpha$',fontsize=24)
	ax_bands.text(13.25,0.02,r'$\beta$',fontsize=24)
	ylims = ax_bands.get_ylim()
	ax_bands.plot([thetaband[0],thetaband[0]],ylims,c='dimgrey',ls=':')
	ax_bands.plot([alphaband[0],alphaband[0]],ylims,c='dimgrey',ls=':')
	ax_bands.plot([alphaband[1],alphaband[1]],ylims,c='dimgrey',ls=':')
	ax_bands.set_ylim(ylims)
	ax_bands.set_xlim(freq0,freq1)
	
	fig_bands.savefig('figs_PSD_SeverityOnly_V2/severity_periodic.png',bbox_inches='tight',dpi=300,transparent=True)
	plt.close()

# Plot spike PSDs
bootCI = True
for cind1, d in enumerate(doses):
	fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(7, 5),sharex=True,sharey=True) # figsize=(17,11)
	inset = ax.inset_axes([.64,.55,.3,.4])
	for cind2, s in enumerate(severities):
		
		if bootCI:
			CI_means_PN = []
			CI_lower_PN = []
			CI_upper_PN = []
			for l in range(0,len(spikes_PN[cind1][cind2][0])):
				x = bs.bootstrap(np.transpose(spikes_PN[cind1][cind2])[l], stat_func=bs_stats.mean, alpha=0.1, num_iterations=50)
				CI_means_PN.append(x.value)
				CI_lower_PN.append(x.lower_bound)
				CI_upper_PN.append(x.upper_bound)
		else:
			CI_means_PN = np.mean(spikes_PN[cind1][cind2],0)
			CI_lower_PN = np.mean(spikes_PN[cind1][cind2],0)-np.std(spikes_PN[cind1][cind2],0)
			CI_upper_PN = np.mean(spikes_PN[cind1][cind2],0)+np.std(spikes_PN[cind1][cind2],0)
		
		f_res = 1/(nperseg*dt/1000)
		f_All = np.arange(0,101,f_res)
		f_PN = np.arange(0,101,f_res)
		
		freq0 = 3
		freq1 = 30
		freq2 = 100
		f1 = np.where(f_All>=freq0)
		f1 = f1[0][0]-1
		f2 = np.where(f_All>=freq1)
		f2 = f2[0][0]+1
		f3 = np.where(f_All>=freq2)
		f3 = f3[0][0]
		
		ax.plot(f_PN[f1:f2], CI_means_PN[f1:f2], color=sm.to_rgba(severities[cind2]),linewidth=3)
		ax.fill_between(f_PN[f1:f2], CI_lower_PN[f1:f2], CI_upper_PN[f1:f2],color=sm.to_rgba(severities[cind2]),alpha=0.3)
		ax.tick_params(axis='x', which='major', bottom=True)
		ax.tick_params(axis='y', which='major', left=True)
		inset.plot(f_PN[:f3], CI_means_PN[:f3], color=sm.to_rgba(severities[cind2]),linewidth=3)
		inset.set_xscale('log')
		inset.set_yscale('log')
		inset.tick_params(axis='x', which='major', bottom=True)
		inset.tick_params(axis='y', which='major', left=True)
		inset.tick_params(axis='x', which='minor', bottom=True)
		inset.tick_params(axis='y', which='minor', left=True)
		inset.xaxis.set_minor_locator(LogLocator(base=10, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
		inset.yaxis.set_minor_locator(LogLocator(base=10, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))
		inset.fill_between(f_PN[:f3], CI_lower_PN[:f3], CI_upper_PN[:f3],color=sm.to_rgba(severities[cind2]),alpha=0.3)
		inset.set_xticks([1,10,freq2])
		inset.xaxis.set_major_formatter(mticker.ScalarFormatter())
		inset.set_xlim(0.3,freq2)
		ax.set_xticks([freq0,5,10,15,20,25,freq1])
		ax.set_xticklabels(['','5','10','15','20','25',str(freq1)])
		ax.set_xlim(freq0,freq1)
		ax.set_xlabel('Frequency (Hz)')
		ax.set_ylabel('Pyr Network PSD (Spikes'+r'$^{2}$'+'/Hz)')
	
	ylims = ax.get_ylim()
	ax.set_ylim(0,3.2*10**-5)
	ax.text(thetaband[0]+(np.diff(thetaband)/2)-0.75,0,r'$\theta$',fontsize=24)
	ax.text(alphaband[0]+(np.diff(alphaband)/2)-0.75,0,r'$\alpha$',fontsize=24)
	ax.text(13.25,0,r'$\beta$',fontsize=24)
	ylims = ax.get_ylim()
	ax.plot([thetaband[0],thetaband[0]],ylims,c='dimgrey',ls=':')
	ax.plot([alphaband[0],alphaband[0]],ylims,c='dimgrey',ls=':')
	ax.plot([alphaband[1],alphaband[1]],ylims,c='dimgrey',ls=':')
	ax.set_ylim(ylims)
	ax.set_xlim(freq0,freq1)
	
	fig.savefig('figs_PSD_SeverityOnly_V2/severity_Spikes_PSD_Boot95CI_PN.png',bbox_inches='tight',dpi=300, transparent=True)
	plt.close()
