#!/bin/sh
sbatch --partition=GPU --gres=gpu:1 --chdir=`pwd` --time=24:00:00 --nodes=1 --mem=120G --job-name=cucnvl cuda_cnvl.sh
sbatch --partition=GPU --gres=gpu:1 --chdir=`pwd` --time=24:00:00 --nodes=1 --mem=120G --job-name=cucnvlop cuda_cnvl_op.sh