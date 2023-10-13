#!/bin/sh
sbatch --partition=Centaurus --chdir=`pwd` --time=01:00:00 --nodes=1 --ntasks-per-node=8 --error='convol_err.txt' --job-name=convl convol.sh
sbatch --partition=Centaurus --chdir=`pwd` --time=01:00:00 --nodes=1 --ntasks-per-node=8 --error='convol_par_err.txt' --job-name=convl_par convol_par.sh