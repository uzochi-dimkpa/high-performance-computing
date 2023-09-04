#!/bin/sh
sbatch --partition=Centaurus --chdir=`pwd` --time=00:10:00 --nodes=1 --tasks-per-node=1 --job-name=advec run_advection.sh
sbatch --partition=Centaurus --chdir=`pwd` --time=00:10:00 --nodes=1 --tasks-per-node=1 --job-name=advec_te run_advection_test.sh
sbatch --partition=Centaurus --chdir=`pwd` --time=00:10:00 --nodes=1 --tasks-per-node=1 --job-name=advec_de run_advection_debug.sh
