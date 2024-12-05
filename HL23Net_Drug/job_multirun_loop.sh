#!/bin/bash --login

#SBATCH --nodes=10
#SBATCH --ntasks-per-node=40
#SBATCH --time=10:00:00
#SBATCH --job-name='Multirun LFPy Circuit'
#SBATCH --account=rrg-etayhay
#SBATCH --mail-type=ALL
#SBATCH --mail-user=agmccrei@gmail.com
#SBATCH -o output1.out
#SBATCH -e error1.out

module load NiaEnv/2018a
module load intel/2018.2
module load intelmpi/2018.2
module load anaconda3/2018.12

conda activate lfpy

unset DISPLAY


for value in {1..20}
do
	mpiexec -n 400 python circuit.py $value
done
