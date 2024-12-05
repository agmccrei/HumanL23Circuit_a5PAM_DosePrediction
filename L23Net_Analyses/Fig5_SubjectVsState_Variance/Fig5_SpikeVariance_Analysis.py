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

base_rates = [[] for _ in N_circuitseedsList]
base_rates_nosilent = [[] for _ in N_circuitseedsList]

for seed_circuit in N_circuitseedsList:
	for seed_state in N_stateseedsList:
		print('Analyzing circuit seed #'+str(seed_circuit)+' and state seed #'+str(seed_state))
		
		temp_s = np.load('Seeds_FullSet/Circuit_output/SPIKES_CircuitSeed' + str(seed_circuit) + '234StimSeed' + str(seed_state) + '.npy',allow_pickle=True)
		
		SPIKES = temp_s.item()
		temp_rn = []
		temp_rn_nosilent = []
		for i in range(0,len(SPIKES['times'][0])):
			scount = SPIKES['times'][0][i][(SPIKES['times'][0][i]>startslice) & (SPIKES['times'][0][i]<=endslice)]
			Hz = scount.size/((endslice-startslice)/1000)
			temp_rn.append(Hz)
			if Hz > 0.2:
				temp_rn_nosilent.append(Hz)
		
		base_rates[seed_circuit-1].append(np.mean(temp_rn))
		base_rates_nosilent[seed_circuit-1].append(np.mean(temp_rn_nosilent))

# Mean & SD state effects per circuit seed
base_rates_circuit = base_rates
base_rates_circuit_m = np.mean([element for sublist in base_rates_circuit for element in sublist]) # np.mean([np.mean(allvals) for allvals in base_rates])
base_rates_circuit_sd = np.std([element for sublist in base_rates_circuit for element in sublist]) #np.mean([np.std(allvals) for allvals in base_rates])
base_rates_circuit_nosilent = base_rates_nosilent
base_rates_circuit_nosilent_m = np.mean([element for sublist in base_rates_circuit_nosilent for element in sublist])
base_rates_circuit_nosilent_sd = np.std([element for sublist in base_rates_circuit_nosilent for element in sublist])

base_rates_circuit_residuals = [[a-np.mean(allvals) for a in allvals] for allvals in base_rates]
base_rates_circuit_residuals_m = np.mean([element for sublist in base_rates_circuit_residuals for element in sublist])
base_rates_circuit_residuals_sd = np.std([element for sublist in base_rates_circuit_residuals for element in sublist])
base_rates_circuit_residuals_nosilent = [[a-np.mean(allvals) for a in allvals] for allvals in base_rates_nosilent]
base_rates_circuit_residuals_nosilent_m = np.mean([element for sublist in base_rates_circuit_residuals_nosilent for element in sublist])
base_rates_circuit_residuals_nosilent_sd = np.std([element for sublist in base_rates_circuit_residuals_nosilent for element in sublist])

base_rates_circuit_zscore = [[(a-np.mean(allvals))/np.std(allvals) for a in allvals] for allvals in base_rates]
base_rates_circuit_zscore_m = np.mean([element for sublist in base_rates_circuit_zscore for element in sublist])
base_rates_circuit_zscore_sd = np.std([element for sublist in base_rates_circuit_zscore for element in sublist])
base_rates_circuit_zscore_nosilent = [[(a-np.mean(allvals))/np.std(allvals) for a in allvals] for allvals in base_rates_nosilent]
base_rates_circuit_zscore_nosilent_m = np.mean([element for sublist in base_rates_circuit_zscore_nosilent for element in sublist])
base_rates_circuit_zscore_nosilent_sd = np.std([element for sublist in base_rates_circuit_zscore_nosilent for element in sublist])

cmap = plt.cm.get_cmap('jet', len(base_rates_circuit_residuals))

fig, ax = plt.subplots(figsize=(5.5,5))
for c,(_,x,y) in enumerate(sorted(zip([np.mean(allvals) for allvals in base_rates],base_rates_circuit_residuals,base_rates_circuit))):
	ax.scatter(x,y,s=8**2,c=np.array([cmap(c)]))
ax.scatter(base_rates_circuit_residuals_m,base_rates_circuit_m,s=8**2,c='k')
ax.errorbar(base_rates_circuit_residuals_m,base_rates_circuit_m,xerr=base_rates_circuit_residuals_sd,yerr=base_rates_circuit_sd,ecolor='k', elinewidth=2, capthick=2, capsize=6, linestyle='')
yscale = ax.get_ylim()
ax.set_xlim(yscale - yscale[0] - (yscale[1]-yscale[0])/2)
ax.set_xlabel('Centered Rate\nAcross States')
ax.set_ylabel('Circuit Rate (Hz)')
#ax.grid()
fig.tight_layout()
fig.savefig('figs_V2/scatters_MeanCenteredVsRate.png',dpi=300,transparent=True)
plt.close()

fig, ax = plt.subplots(figsize=(6,5))
for c,(_,x,y) in enumerate(sorted(zip([np.mean(allvals) for allvals in base_rates_circuit_nosilent],base_rates_circuit_residuals_nosilent,base_rates_circuit_nosilent))):
	ax.scatter(x,y,s=8**2,c=np.array([cmap(c)]))
ax.scatter(base_rates_circuit_residuals_nosilent_m,base_rates_circuit_nosilent_m,s=8**2,c='k')
ax.errorbar(base_rates_circuit_residuals_nosilent_m,base_rates_circuit_nosilent_m,xerr=base_rates_circuit_residuals_nosilent_sd,yerr=base_rates_circuit_nosilent_sd,ecolor='k', elinewidth=2, capthick=2, capsize=6, linestyle='')
yscale = ax.get_ylim()
ax.set_xlim(yscale - yscale[0] - (yscale[1]-yscale[0])/2)
ax.set_xlabel('Centered Rate\nAcross States')
ax.set_ylabel('Circuit Rate (Hz)')
ax.grid()
fig.tight_layout()
fig.savefig('figs_V2/scatters_MeanCenteredVsRate_nosilent.png',dpi=300,transparent=True)
plt.close()

fig, ax = plt.subplots(figsize=(5,5))
for c,(_,x,y) in enumerate(sorted(zip([np.mean(allvals) for allvals in base_rates],base_rates_circuit_zscore,base_rates_circuit))):
	ax.scatter(x,y,s=8**2,c=np.array([cmap(c)]))
ax.scatter(base_rates_circuit_zscore_m,base_rates_circuit_m,s=8**2,c='k')
ax.errorbar(base_rates_circuit_zscore_m,base_rates_circuit_m,xerr=base_rates_circuit_zscore_sd,yerr=base_rates_circuit_sd,ecolor='k', elinewidth=2, capthick=2, capsize=6, linestyle='')
ax.set_xlabel('Z-scored Rate (SD)')
ax.set_ylabel('Circuit Rate (Hz)')
ax.grid()
fig.tight_layout()
fig.savefig('figs_V2/scatters_ZscoreVsRate.png',dpi=300,transparent=True)
plt.close()

fig, ax = plt.subplots(figsize=(5,5))
for c,(_,x,y) in enumerate(sorted(zip([np.mean(allvals) for allvals in base_rates_circuit_nosilent],base_rates_circuit_zscore_nosilent,base_rates_circuit_nosilent))):
	ax.scatter(x,y,s=8**2,c=np.array([cmap(c)]))
ax.scatter(base_rates_circuit_zscore_nosilent_m,base_rates_circuit_nosilent_m,s=8**2,c='k')
ax.errorbar(base_rates_circuit_zscore_nosilent_m,base_rates_circuit_nosilent_m,xerr=base_rates_circuit_zscore_nosilent_sd,yerr=base_rates_circuit_nosilent_sd,ecolor='k', elinewidth=2, capthick=2, capsize=6, linestyle='')
ax.set_xlabel('Z-scored Rate (SD)')
ax.set_ylabel('Circuit Rate (Hz)')
ax.grid()
fig.tight_layout()
fig.savefig('figs_V2/scatters_ZscoreVsRate_nosilent.png',dpi=300,transparent=True)
plt.close()

# Mean & SD circuit effects per state seed
base_rates_state_m = [np.mean(allvals) for allvals in np.transpose(base_rates)]
base_rates_state_sd = [np.std(allvals) for allvals in np.transpose(base_rates)]

base_rates_circuit_m = [np.mean(allvals) for allvals in base_rates]
base_rates_circuit_sd = [np.std(allvals) for allvals in base_rates]

base_rates_state_variance_m = np.mean(base_rates_state_sd)
base_rates_circuit_variance_m = np.mean(base_rates_circuit_sd)
base_rates_state_variance_sd = np.std(base_rates_state_sd)
base_rates_circuit_variance_sd = np.std(base_rates_circuit_sd)

# Plot variance across states vs variance across circuits
font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':20})

fig, ax = plt.subplots(figsize=(3.4,4.25))
xvals = [0,1]
ax.bar(xvals,[base_rates_state_variance_m,base_rates_circuit_variance_m],yerr=[base_rates_state_variance_sd,base_rates_circuit_variance_sd],width=0.7,color='grey',edgecolor='k',linewidth=3,error_kw=dict(lw=3, capsize=5, capthick=3))
ax.set_xticks(xvals)
ax.set_xticklabels(['Circuit','State'], rotation=45, ha='center')
ax.set_ylabel('SD (Hz)')
ax.set_xlim(-0.75,1.75)
#ax.yaxis.tick_right()
#ax.yaxis.set_label_position("right")
fig.tight_layout()
fig.savefig('figs_V2/scatters_Variance.png',dpi=300,transparent=True)
plt.close()

g1_name = 'Circuit Variance'
g1 = base_rates_state_sd
g2_name = 'State Variance'
g2 = base_rates_circuit_sd

print(g1_name+' mean = '+str(np.mean(g1)) + '\u00B1' + str(np.std(g1)))
print(g2_name+' mean = '+str(np.mean(g2)) + '\u00B1' + str(np.std(g2)))
print('Cohens D = ' + str(cohen_d(g1,g2)))
if st.levene(g1,g2)[1] >= 0.05: print('Independent-Sample T-Test')
else: print('Welsh T-Test')
print('t-test p-value = ' + str(st.ttest_ind(g1,g2,equal_var=True if st.levene(g1,g2)[1] >= 0.05 else False)[1]))
print('t-test t-stat = ' + str(st.ttest_ind(g1,g2,equal_var=True if st.levene(g1,g2)[1] >= 0.05 else False)[0]))
