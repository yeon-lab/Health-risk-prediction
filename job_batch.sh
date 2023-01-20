#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=8 --gpus-per-node=1 --gpu_cmode=shared

module load python
source activate local

python train.py --config 'config.json'
