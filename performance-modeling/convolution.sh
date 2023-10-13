#!/bin/sh
sbatch --partition=Centaurus --chdir=`pwd` --time=01:00:00 --nodes=1 --ntasks-per-node=8 --job-name=convl run convol.sh
sbatch --partition=Centaurus --chdir=`pwd` --time=01:00:00 --nodes=1 --ntasks-per-node=8 --job-name=convl_par run convol_par.sh