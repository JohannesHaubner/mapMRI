#!/bin/bash
#SBATCH --job-name=register
#SBATCH --cpus-per-task=1 --nodes=1
## SBATCH --partition=milanq
#SBATCH --partition=defq
## defq also possible

#SBATCH --time 5-24 ##--time=99:00:00
#SBATCH -o /home/bastian/Oscar-Image-Registration-via-Transport-Equation/slurm/%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=bastian@simula.no
#module load gcc
#module load anaconda3/x86_64
#export LD_LIBRARY_PATH=/home/bastian/.conda/envs/mri_inverse/lib
# conda init bash

eval "$(conda shell.bash hook)"
conda activate mri_inverse

SPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation



set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error



outfolder=${SPATH}/slurm/${SLURM_JOB_ID}

srun -n 1 python3 ${SPATH}/Optimize3d.py > ${outfolder}_log_python_srun.txt
 
