# Human L2/3 Cortical Circuit Model for Therapeutic Dose Prediction of New Pharmacology in Depression --- Guet-McCreight-et-al.-2023
===================================================================================================================================================================
Author: Alexandre Guet-McCreight

This is the readme for the model associated with the paper:

Guet-McCreight A, Mazza F, Prevot TD, Sibille E, Hay E (2024) Therapeutic dose prediction of α5-GABA receptor modulation from simulated EEG of depression severity

This code is part of a provisional US patent. Name: EEG biomarkers for Alpha5-PAM therapy (US provisional patent no. 63/382,577).

Network Simulations:
Simulation code associated with the L2/3 circuit used throughout the manuscript is in the /L23Net_Drug/ directory. Note that this circuit model is adapted from https://doi.org/10.5281/zenodo.10497761.

To run simulations, install all of the necessary python modules (see lfpy_env.yml), compile the mod files within the mod folder, and submit the simulations in parallel (e.g., see job_multirun_loop.sh). 

In job_multirun_loop.sh, the number at the end of the mpiexec command (see below - 1234) controls the random seed used for both the circuit variance (i.e., connection matrix, synapse placement, etc.) and the stimulus variance (i.e. Ornstein Uhlenbeck noise and stimulus presynaptic spike train timing).

mpiexec -n 400 python circuit.py 1234

Here are some of the different configurations of parameters in circuit.py that we change to look at different conditions and levels of analysis.

EEG simulations: 
tstop = 25000.

Healthy Parameters:
MDD = 0 to 0.4
DRUG_a5PAM = 0 to 1.5

Analysis and Dose Prediction:
All code used for analysis and dose prediction of the circuit simulation results is found in the /L23Net_Analyses/ directory. Creation of subfolders for plots and analysis results may be necessary to run this code.

This folder also includes spike simulation outputs and EEG power spectra (generated using Plot_PSD_SavePSD2NPY_nperseg_80000.py - see "Network Simulations" above for instructions on generating simulated EEG).

This folder also includes a subfolder (/Fig5_SubjectVsState_Variance/) containing code for a subject vs state variance analysis. Circuit simulation code in this subfolder is slightly different from the main simulation code in that the random seeds are controlled separately for subject circuit variance (i.e., connection matrix, synapse placement, etc.) vs state variance (i.e. Ornstein Uhlenbeck noise and stimulus presynaptic spike train timing). As with other folders, for this analysis we include the appropriate simulation spiking and power spectral density results in the /Seeds_FullSet/ and /Saved_PSDs_AllSeeds/ folders, respectively.
