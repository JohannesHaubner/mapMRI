#!/bin/bash
#SBATCH --job-name=transport
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#################################################################
##SBATCH --partition=milanq
#SBATCH --partition=fpgaq
## SBATCH --partition=defq
## SBATCH --partition=slowq
## SBATCH --partition=rome16q
#################################################################

#SBATCH --time 0-2 ##--time=99:00:00
#SBATCH -o /home/bastian/Oscar-Image-Registration-via-Transport-Equation/slurm/transportslurm/%j.out
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
SPATH=${GPATH}scripts/postprocessing
LOGOUTFILE=${GPATH}slurm/transportslurm/${SLURM_JOB_ID}_log_python_srun.txt

cd $SPATH

FOLDER=croppedmriregistration_outputs
JOBNAME=OCDA10.0LBFGS5

# FOLDER=mriregistration_outputs
# JOBNAME=E100A0.001LBFGS100
# OCDA0.001LBFGS1000


srun -n 8 python3 -u transform.py --mapping_only --folder ${FOLDER} --job ${JOBNAME} --velocityfilename opts/opt_phi_6.h5 --readname "function" > ${LOGOUTFILE}

echo "Computed mapping, now running transform with only one core"

srun -n 1 python3 -u transform.py --folder ${FOLDER} --job ${JOBNAME} --velocityfilename opts/opt_phi_6.h5 --readname "function" >> ${LOGOUTFILE}

echo "success"