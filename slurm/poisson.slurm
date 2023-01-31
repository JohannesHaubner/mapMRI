#!/bin/bash
#SBATCH --job-name=quicktest
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=128
#################################################################
## SBATCH --partition=milanq
#SBATCH --partition=fpgaq
## SBATCH --partition=defq
## SBATCH --exclusive
## SBATCH --reservation=bastian_39
##SBATCH --partition=habanaq
#################################################################
#SBATCH --time 0-8 ##--time=99:00:00
#SBATCH -o /home/bastian/D1/registration/poissonslurm/%j.out
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
LOGOUTFILE=/home/bastian/D1/registration/poissonslurm/${SLURM_JOB_ID}_log_python_srun.txt

NP=16

meshn=100
LBFGS=100
NCOR=2


echo "NTASKS="$NP
echo "meshn="$meshn
echo "lbfgs="$LBFGS
echo "NCOR="$NCOR

srun -n $NP python3 -u ${SPATH}/Poisson.py --lbfgs_max_iterations ${LBFGS} --maxcor ${NCOR} --meshn ${meshn} > ${LOGOUTFILE}



echo "success"