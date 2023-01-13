#!/bin/bash
#SBATCH --job-name=mriregister
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=40
#################################################################
## SBATCH --partition=milanq
## SBATCH --partition=fpgaq
#SBATCH --partition=defq
## SBATCH --partition=slowq
## SBATCH --partition=rome16q
#################################################################
#SBATCH --time 13-00:00:0
##SBATCH --time 6-24 ##--time=99:00:00
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



ITERS=$1
ALPHA=$2
LBFGS=100

echo "ITERS=" ${ITERS}
echo "ALPHA=" ${ALPHA}

OUTFOLDERNAME=None




OUTPUTDIR=/home/bastian/D1/registration/croppedmriregistration_outputs/
IMG1=/home/bastian/D1/registration/mri2fem-dataset/processed/cropped/cropped_abbytoernie.mgz
IMG2=/home/bastian/D1/registration/mri2fem-dataset/processed/cropped/cropped_ernie_brain.mgz
# OUTFOLDERNAME=RS1e-3

mkdir -pv ${OUTPUTDIR}
echo "OUTPUTDIR=" $OUTPUTDIR
echo "OUTFOLDERNAME=" $OUTFOLDERNAME
echo "IMG1=" ${IMG1}
echo "IMG2=" ${IMG2}

srun -n 40 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --logfile ${LOGOUTFILE} --output_dir ${OUTPUTDIR} \
--input ${IMG1} --target ${IMG2} --max_timesteps ${ITERS} --alpha ${ALPHA} --lbfgs_max_iterations ${LBFGS} \
--timestepping explicitEuler --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}

# --starting_guess /home/bastian/D1/registration/mriregistration_outputs/E100A0.001LBFGS100/Control.hdf --readname "-1" --interpolate \


# OUTPUTDIR=/home/bastian/D1/registration/mriregistration_outputs/
# DPATH=/home/bastian/D1/registration
# DATAPATH=${DPATH}/mri2fem-dataset/processed/coarsecropped/
# IMG1=${DATAPATH}coarsenedabbytoernie.mgz
# IMG2=${DATAPATH}coarsenedernie_brain.mgz


# mkdir -pv ${OUTPUTDIR}
# echo "OUTPUTDIR=" $OUTPUTDIR
# echo "OUTFOLDERNAME=" $OUTFOLDERNAME
# echo "IMG1=" ${IMG1}
# echo "IMG2=" ${IMG2}

# srun -n 30 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --logfile ${LOGOUTFILE} --output_dir ${OUTPUTDIR} \
# --input ${IMG1} --target ${IMG2} --max_timesteps ${ITERS} --alpha ${ALPHA} --lbfgs_max_iterations ${LBFGS} \
# --starting_guess /home/bastian/D1/registration/mriregistration_outputs/E100A0.001LBFGS100/Control.hdf --readname "-1" \
# --timestepping explicitEuler --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}
# #--state_functiondegree 0  --nosmoothen  --nosmoothen
# #  --timestepping explicitEuler --sol


echo "success"