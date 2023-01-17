#!/bin/bash
#SBATCH --job-name=mriregister
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#################################################################
## SBATCH --partition=milanq
#SBATCH --partition=fpgaq
## SBATCH --partition=defq
## SBATCH --partition=slowq
## SBATCH --partition=rome16q
#################################################################
#SBATCH --time 1-00:00:0
##SBATCH --time 6-24 ##--time=99:00:00
#SBATCH -o /home/bastian/D1/registration/mrislurm/%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=bastian@simula.no
#module load gcc
#module load anaconda3/x86_64
#export LD_LIBRARY_PATH=/home/bastian/.conda/envs/mri_inverse/lib
# conda init bash

eval "$(conda shell.bash hook)"
conda activate mri_inverse

# module load ipopt-3.13.3

set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

echo "starting to run"


#######################################################
## EXECUTE AFTER ANOTHER JOB IS DONE:
## sbatch --dependency=afterany: submit.slurm
#######################################################


GPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/
SPATH=${GPATH}scripts/optimization
LOGOUTFILE=./${SLURM_JOB_ID}_log_python_srun.txt


srun -n 1 python3 -u test.py 2 > ${LOGOUTFILE}

echo "success"