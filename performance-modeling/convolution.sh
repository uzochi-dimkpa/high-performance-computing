#!/bin/sh
sbatch --partition=Centaurus --chdir=`pwd` --time=24:00:00 --nodes=1 --ntasks-per-node=16 --mem=120G --mail-type=END,FAIL,REQUEUE,TIME_LIMIT --mail-user=udimkpa@uncc.edu --job-name=convl convol.sh
sbatch --partition=Centaurus --chdir=`pwd` --time=24:00:00 --nodes=1 --ntasks-per-node=16 --mem=120G --mail-type=END,FAIL,REQUEUE,TIME_LIMIT --mail-user=udimkpa@uncc.edu --job-name=convlpar convol_par.sh