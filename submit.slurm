#!/bin/bash
#SBATCH --job-name=register
#SBATCH --cpus-per-task=1 --nodes=4
## SBATCH --partition=milanq
#SBATCH --partition=fpgaq
## SBATCH --partition=defq
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

# module load ipopt-3.13.3

SPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation



set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error



outfolder=${SPATH}/slurm/${SLURM_JOB_ID}

OUTFOLDERNAME=explicitEuler

echo "starting to run"

srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --timestepping explicitEuler --use_krylov_solver > ${outfolder}_log_python_srun.txt

# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder output_coarsened_mri_ipopt > ${outfolder}_log_python_srun.txt
 
echo "success"