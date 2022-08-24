#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1 --ntasks-per-node=8 --gpus-per-node=1 --gpu_cmode=shared
#SBATCH --job-name=dipole
#SBATCH --account=PCON0023
#SBATCH --exclude=p0240

module load python
source activate local

python train_dipole.py --config 'config_dipole.json' --np_data_dir "./data_npy_inpat/"