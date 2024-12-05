#!/bin/bash --login

#SBATCH --nodes=10
#SBATCH --ntasks-per-node=40
#SBATCH --time=10:00:00
#SBATCH --job-name='Multirun LFPy Circuit'
#SBATCH --account=rrg-etayhay
#SBATCH --mail-type=ALL
#SBATCH --mail-user=agmccrei@gmail.com
#SBATCH -o output_1.out
#SBATCH -e error_1.out

module load NiaEnv/2018a
module load intel/2018.2
module load intelmpi/2018.2
module load anaconda3/2018.12

conda activate lfpy

unset DISPLAY

mpiexec -n 400 python circuit.py 6234 1
mpiexec -n 400 python circuit.py 6234 2
mpiexec -n 400 python circuit.py 6234 3
mpiexec -n 400 python circuit.py 6234 4
mpiexec -n 400 python circuit.py 6234 5
mpiexec -n 400 python circuit.py 6234 6
mpiexec -n 400 python circuit.py 6234 7
mpiexec -n 400 python circuit.py 6234 8
mpiexec -n 400 python circuit.py 6234 9
mpiexec -n 400 python circuit.py 6234 10
