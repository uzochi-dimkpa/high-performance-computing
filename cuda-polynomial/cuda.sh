#!/bin/sh
sbatch --partition=GPU --gres=gpu:1 --chdir=`pwd` --time=00:10:00 --nodes=1 --mem=120G --job-name=cudapoly cuda_poly.sh
### --ntasks-per-node=16 --mem=120G --mail-type=END,FAIL,REQUEUE,TIME_LIMIT --mail-user=udimkpa@uncc.edu
