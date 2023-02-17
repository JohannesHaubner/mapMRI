#!/bin/bash
#SBATCH --job-name=mriregister
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=256
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#################################################################
#SBATCH --partition=milanq
## milanq has 256 cpus per node
#SBATCH --partition=fpgaq
# fpgaq has 96 cpus per onde
## SBATCH --partition=defq
## has 128 cpus per node
## SBATCH --partition=habanaq
#################################################################
#SBATCH --time 3-00:00:0
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

NP=64

echo "NP=" ${NP}

# ITERS=$1
# ALPHA=$2
ITERS=50
ALPHA=1e2
LBFGS=16

echo "ITERS=" ${ITERS}
echo "ALPHA=" ${ALPHA}

OUTFOLDERNAME=None

OUTPUTDIR=/home/bastian/D1/registration/normalized-outputs/${SLURM_JOB_ID}/
IMG1=/home/bastian/D1/registration/mri2fem-dataset/normalized/cropped/cropped_abbytoernie_nyul.mgz
IMG2=/home/bastian/D1/registration/mri2fem-dataset/normalized/cropped/cropped_ernie_brain_nyul.mgz

# OUTPUTDIR=/home/bastian/D1/registration/hubertest_coarse/${SLURM_JOB_ID}/
# IMG1=/home/bastian/D1/registration/mri2fem-dataset/processed/coarsecropped/coarsenedabbytoernie.mgz
# IMG2=/home/bastian/D1/registration/mri2fem-dataset/processed/coarsecropped/coarsenedernie_brain.mgz


echo "OUTPUTDIR=" $OUTPUTDIR
echo "OUTFOLDERNAME=" $OUTFOLDERNAME
echo "IMG1=" ${IMG1}
echo "IMG2=" ${IMG2}


srun -n ${NP} python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --logfile ${LOGOUTFILE} --output_dir ${OUTPUTDIR} \
--input ${IMG1} --target ${IMG2} --max_timesteps ${ITERS} --alpha ${ALPHA} --lbfgs_max_iterations ${LBFGS} \
--huber --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}
 
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
