#!/bin/bash
#SBATCH --job-name=mriregister
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#################################################################
## SBATCH --partition=milanq
#SBATCH --partition=fpgaq
## SBATCH --partition=defq
## SBATCH --partition=habanaq
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
LOGOUTFILE=/home/bastian/D1/registration/mrislurm/${SLURM_JOB_ID}_log_python_srun.txt


# echo "Running dijitso clean"
# dijitso clean

NP=32

echo "NP=" ${NP}

# ITERS=$1
# ALPHA=$2
ITERS=10
ALPHA=1e0
LBFGS=50

echo "ITERS=" ${ITERS}
echo "ALPHA=" ${ALPHA}

OUTFOLDERNAME=None

# OUTFOLDERNAME=NP${NP}


# OUTPUTDIR=/home/bastian/D1/registration/affine_croppedmriregistration_outputs/${SLURM_JOB_ID}/
# IMG1=/home/bastian/D1/registration/mri2fem-dataset/processed/affine_registered/cropped_abbytoernie_affine.mgz
# IMG2=/home/bastian/D1/registration/mri2fem-dataset/processed/affine_registered/cropped_ernie_brain.mgz

# OUTPUTDIR=/home/bastian/D1/registration/cropped-outputs/${SLURM_JOB_ID}/
# IMG1=/home/bastian/D1/registration/mri2fem-dataset/processed/cropped/cropped_abbytoernie.mgz
# IMG2=/home/bastian/D1/registration/mri2fem-dataset/processed/cropped/cropped_ernie_brain.mgz

# OUTPUTDIR=/home/bastian/D1/registration/coarsecropped-outputs/${SLURM_JOB_ID}/
OUTPUTDIR=/home/bastian/D1/registration/hubertest_coarse/${SLURM_JOB_ID}/
IMG1=/home/bastian/D1/registration/mri2fem-dataset/processed/coarsecropped/coarsenedabbytoernie.mgz
IMG2=/home/bastian/D1/registration/mri2fem-dataset/processed/coarsecropped/coarsenedernie_brain.mgz




# dijitso show
# dijitso config

# mkdir -pv ${OUTPUTDIR}

# DIJITSODIR=${OUTPUTDIR}.cache/dijitso
# mkdir -pv ${DIJITSODIR}
# export DIJITSO_CACHE_DIR=${DIJITSODIR}

# echo "Setting new dijitso cache dir"
# dijitso show
# dijitso config


echo "OUTPUTDIR=" $OUTPUTDIR
echo "OUTFOLDERNAME=" $OUTFOLDERNAME
echo "IMG1=" ${IMG1}
echo "IMG2=" ${IMG2}


srun -n ${NP} python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --logfile ${LOGOUTFILE} --output_dir ${OUTPUTDIR} \
--input ${IMG1} --target ${IMG2} --max_timesteps ${ITERS} --alpha ${ALPHA} --lbfgs_max_iterations ${LBFGS} \
--slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}
 
# --huber

# --starting_guess /D1/registration/affine_croppedmriregistration_outputs/441848/E100A0.01LBFGS10C5/Control.hdf --readname -1 \

# --tukey --tukey_c 5 
# --starting_state  /home/bastian/D1/registration/affine_croppedmriregistration_outputs/441870/E100A0.01LBFGS10//State_checkpoint.xdmf \
# --starting_guess /D1/registration/affine_croppedmriregistration_outputs/441848/E100A0.01LBFGS10C5/Control.hdf --readname -1
# --starting_state /home/bastian/D1/registration/affine_croppedmriregistration_outputs/441849/E100A0.01LBFGS10/State_checkpoint.xdmf \
# --tukey --tukey_c 1 
# --memdebug --not_store_solver 
#  --reassign --memdebug 
#--not_store_solver --padding 0 
#--not_store_solver 
# --not_store_solver 
# --solver cg 
########## --preconditioning none 
########## ADD --exclusive IF RUNNING WITHOUT PRECONDITIONING
#########
# --nosmoothen
# #--preconditioning none 
# --starting_state ${GUESSPATH}/State_checkpoint.xdmf \

#
echo "success"
