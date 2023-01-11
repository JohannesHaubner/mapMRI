#!/bin/bash
#SBATCH --job-name=ocdregister
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=12
#################################################################
## SBATCH --partition=milanq
## SBATCH --partition=fpgaq
#SBATCH --partition=defq
## SBATCH --partition=slowq
## SBATCH --partition=rome16q
#################################################################

#SBATCH --time 6-24 ##--time=99:00:00
#SBATCH -o /home/bastian/Oscar-Image-Registration-via-Transport-Equation/slurm/mrislurm/%j.out
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
LOGOUTFILE=${GPATH}slurm/mrislurm/${SLURM_JOB_ID}_log_python_srun.txt



ITERS=1
ALPHA=$1
LBFGS=100

echo "ITERS=" ${ITERS}
echo "ALPHA=" ${ALPHA}

OUTFOLDERNAME=None

OUTPUTDIR=/home/bastian/D1/registration/mriregistration_outputs/
# OUTPUTDIR=/home/bastian/D1/registration/readtest/


DPATH=/home/bastian/D1/registration
DATAPATH=${DPATH}/mri2fem-dataset/processed/coarsecropped/
IMG1=${DATAPATH}coarsenedabbytoernie.mgz
IMG2=${DATAPATH}coarsenedernie_brain.mgz
# --starting_guess /home/bastian/D1/registration/mriregistration_outputs/OCDA0.0001LBFGS1000/opts/opt_phi_529.h5 --readname "function" \
# OUTFOLDERNAME=RS1e-4
OUTFOLDERNAME=RSNone

# OUTPUTDIR=/home/bastian/D1/registration/croppedmriregistration_outputs/
# OUTFOLDERNAME=RS10
# IMG1=/home/bastian/D1/registration/mri2fem-dataset/processed/cropped/cropped_abbytoernie.mgz
# IMG2=/home/bastian/D1/registration/mri2fem-dataset/processed/cropped/cropped_ernie_brain.mgz
# --starting_guess /home/bastian/D1/registration/croppedmriregistration_outputs/OCDA10.0LBFGS5/opts/opt_phi_6.h5 --readname "function" \

mkdir -pv ${OUTPUTDIR}
echo "OUTPUTDIR=" $OUTPUTDIR
echo "OUTFOLDERNAME=" $OUTFOLDERNAME
echo "IMG1=" ${IMG1}
echo "IMG2=" ${IMG2}


srun -n 12 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
--input ${IMG1} --target ${IMG2} --max_timesteps ${ITERS} --alpha ${ALPHA} --logfile ${LOGOUTFILE} \
--ocd --lbfgs_max_iterations ${LBFGS} --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}

echo "success"