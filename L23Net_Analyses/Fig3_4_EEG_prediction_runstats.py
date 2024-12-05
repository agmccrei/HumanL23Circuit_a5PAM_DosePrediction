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

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc('legend',**{'fontsize':26})

def cohen_d(y,x):
	nx = len(x)
	ny = len(y)
	dof = nx + ny - 2
	return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

#fname = 'accuracy_pmDose_allvalues.npy'
#fname = 'accuracy_pmDose_allvalues_above.npy'
#fname = 'accuracy_pmDose_allvalues_below.npy'
directories = ['figs_ManualLinearRegression_V13/accuracy_pmDose_allvalues.npy',
				'figs_ManualLinearRegression_V13_SVM/accuracy_pmDose_allvalues.npy']
all_data = [np.load(d) for d in directories]

g1_name = 'MV linear regression'
g1 = all_data[0][3]
g2_name = 'SVM linear regression'
g2 = all_data[1][3]

print(g1_name+' mean = '+str(np.mean(g1)) + '\u00B1' + str(np.std(g1)))
print(g2_name+' mean = '+str(np.mean(g2)) + '\u00B1' + str(np.std(g2)))
print('Cohens D = ' + str(cohen_d(g1,g2)))
if st.levene(g1,g2)[1] >=0.05:
	print('Levene p-value = ' + str(st.levene(g1,g2)[1]))
	print('Performing Independent T-Test')
elif st.levene(g1,g2)[1] < 0.05:
	print('Levene p-value = ' + str(st.levene(g1,g2)[1]))
	print('Performing Welchs T-Test')
print('t-test p-value = ' + str(st.ttest_ind(g1,g2,equal_var=True if st.levene(g1,g2)[1] >=0.05 else False)[1]))
print('t-test t-stat = ' + str(st.ttest_ind(g1,g2,equal_var=True if st.levene(g1,g2)[1] >=0.05 else False)[0]))
