#================================================================================
#= Import
#================================================================================
import os
import time
tic = time.perf_counter()
from os.path import join
import sys
import zipfile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy
from scipy import signal as ss
from scipy import stats as st
from mpi4py import MPI
import math
import neuron
from neuron import h, gui
import LFPy
from LFPy import NetworkCell, Network, Synapse, RecExtElectrode, StimIntElectrode
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import itertools

#================================================================================
#= Controls
#================================================================================

plotrasterandrates = True
plotsomavs = True # Specify cell indices to plot in 'cell_indices_to_plot' - Note: plotting too many cells can randomly cause TCP connection errors

#===============================
#= Analysis Parameters
#===============================
transient = 1000 #used for plotting and analysis

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 14}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':16})
#===============================


# Plot spike time histograms
def plot_spiketimehists(SPIKES,network,gstart=transient,gstop=network.tstop,stimtime=0,binsize=10):
	colors = ['dimgray', 'crimson', 'green', 'darkorange']
	numbins = int((gstop - gstart)/binsize)
	fig, axarr = plt.subplots(len(colors),1)
	for i, pop in enumerate(network.populations):
		popspikes = list(itertools.chain.from_iterable(SPIKES['times'][i]))
		popspikes = [i2-stimtime for i2 in popspikes]
		axarr[i].hist(popspikes,bins=numbins,color=colors[i],linewidth=0,edgecolor='none',range=(gstart,gstop))
		axarr[i].set_xlim(gstart,gstop)
		if i < len(colors)-1:
			axarr[i].set_xticks([])
	axarr[-1:][0].set_xlabel('Time (ms)')
	
	return fig

# Collect Somatic Voltages Across Ranks
def somavCollect(network,cellindices,RANK,SIZE,COMM):
	if RANK == 0:
		volts = []
		gids2 = []
		for i, pop in enumerate(network.populations):
			svolts = []
			sgids = []
			for gid, cell in zip(network.populations[pop].gids, network.populations[pop].cells):
				if gid in cellindices:
					svolts.append(cell.somav)
					sgids.append(gid)

			volts.append([])
			gids2.append([])
			volts[i] += svolts
			gids2[i] += sgids

			for j in range(1, SIZE):
				volts[i] += COMM.recv(source=j, tag=15)
				gids2[i] += COMM.recv(source=j, tag=16)
	else:
		volts = None
		gids2 = None
		for i, pop in enumerate(network.populations):
			svolts = []
			sgids = []
			for gid, cell in zip(network.populations[pop].gids, network.populations[pop].cells):
				if gid in cellindices:
					svolts.append(cell.somav)
					sgids.append(gid)
			COMM.send(svolts, dest=0, tag=15)
			COMM.send(sgids, dest=0, tag=16)

	return dict(volts=volts, gids2=gids2)

# Plot somatic voltages for each population
def plot_somavs(network,VOLTAGES,gstart=transient,gstop=network.tstop,tstim=0):
	tvec = np.arange(network.tstop/network.dt+1)*network.dt-tstim
	fig = plt.figure(figsize=(10,5))
	cls = ['black','crimson','green','darkorange']
	for i, pop in enumerate(network.populations):
		for v in range(0,len(VOLTAGES['volts'][i])):
			ax = plt.subplot2grid((len(VOLTAGES['volts']), len(VOLTAGES['volts'][i])), (i, v), rowspan=1, colspan=1, frameon=False)
			ax.plot(tvec,VOLTAGES['volts'][i][v], c=cls[i])
			ax.set_xlim(gstart,gstop)
			ax.set_ylim(-85,45)
			if i < len(VOLTAGES['volts'])-1:
				ax.set_xticks([])
			if v > 0:
				ax.set_yticks([])
	
	return fig

# Run Plot Functions
if plotsomavs:
	N_HL23PYR = circuit_params['SING_CELL_PARAM'].at['cell_num','HL23PYR']
	N_HL23SST = circuit_params['SING_CELL_PARAM'].at['cell_num','HL23SST']
	N_HL23PV = circuit_params['SING_CELL_PARAM'].at['cell_num','HL23PV']
	N_HL23VIP = circuit_params['SING_CELL_PARAM'].at['cell_num','HL23VIP']
	cell_indices_to_plot = [0, N_HL23PYR, N_HL23PYR+N_HL23SST, N_HL23PYR+N_HL23SST+N_HL23PV]
	VOLTAGES = somavCollect(network,cell_indices_to_plot,RANK,SIZE,COMM)

if RANK ==0:
	if plotrasterandrates:
		fig = plot_spiketimehists(SPIKES,network,gstart=-200,gstop=200,stimtime=2000)
		fig.savefig(os.path.join(OUTPUTPATH,'spiketimesStim_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig = plot_spiketimehists(SPIKES,network,gstart=-100,gstop=100,stimtime=2000,binsize=3)
		fig.savefig(os.path.join(OUTPUTPATH,'spiketimesStim3msBins_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
	if plotsomavs:
		fig = plot_somavs(network,VOLTAGES)
		fig.savefig(os.path.join(OUTPUTPATH,'somav_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)
		fig = plot_somavs(network,VOLTAGES,gstart=-200,gstop=200,tstim=2000)
		fig.savefig(os.path.join(OUTPUTPATH,'somavStim_'+str(GLOBALSEED)+'StimSeed'+str(STIMSEED)),bbox_inches='tight', dpi=300, transparent=True)

