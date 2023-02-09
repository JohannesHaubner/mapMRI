#!/bin/bash
#SBATCH --job-name=mriregister
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=40
#SBATCH --nodes=1
## SBATCH --exclusive
### =user
## SBATCH --ntasks-per-node=40
#################################################################
##SBATCH --partition=milanq
##SBATCH --partition=fpgaq
##
#SBATCH --partition=defq
##SBATCH --reservation=bastian_39
##
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



# ITERS=$1
# ALPHA=$2
ITERS=100
ALPHA=1e-1
LBFGS=40

echo "ITERS=" ${ITERS}
echo "ALPHA=" ${ALPHA}

OUTFOLDERNAME=None
OUTFOLDERNAME=None


# OUTPUTDIR=/home/bastian/D1/registration/affine-rotated-outputs_milanq/
# # OUTPUTDIR=/home/bastian/D1/registration/affine-rotated-outputs_restart/
# IMG1=/home/bastian/D1/registration/mri2fem-dataset/processed/affine-rotated/abby.mgz
# IMG2=/home/bastian/D1/registration/mri2fem-dataset/processed/affine-rotated/ernie.mgz

# # OUTPUTDIR=/home/bastian/D1/registration/affine_croppedmriregistration_outputs/tukeyC1/
# # IMG1=/home/bastian/D1/registration/mri2fem-dataset/processed/affine_registered/cropped_abbytoernie_affine.mgz
# # IMG2=/home/bastian/D1/registration/mri2fem-dataset/processed/affine_registered/cropped_ernie_brain.mgz

# OUTPUTDIR=/home/bastian/D1/registration/cropped-outputs/${SLURM_JOB_ID}/
# IMG1=/home/bastian/D1/registration/mri2fem-dataset/processed/cropped/cropped_abbytoernie.mgz
# IMG2=/home/bastian/D1/registration/mri2fem-dataset/processed/cropped/cropped_ernie_brain.mgz

OUTPUTDIR=/home/bastian/D1/registration/coarsecropped-outputs/${SLURM_JOB_ID}/
IMG1=/home/bastian/D1/registration/mri2fem-dataset/processed/coarsecropped/coarsenedabbytoernie.mgz
IMG2=/home/bastian/D1/registration/mri2fem-dataset/processed/coarsecropped/coarsenedernie_brain.mgz




dijitso show
dijitso config

mkdir -pv ${OUTPUTDIR}

DIJITSODIR=${OUTPUTDIR}.cache/dijitso
mkdir -pv ${DIJITSODIR}
export DIJITSO_CACHE_DIR=${DIJITSODIR}

echo "Setting new dijitso cache dir"
dijitso show
dijitso config





echo "OUTPUTDIR=" $OUTPUTDIR
echo "OUTFOLDERNAME=" $OUTFOLDERNAME
echo "IMG1=" ${IMG1}
echo "IMG2=" ${IMG2}


srun -n 40 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --logfile ${LOGOUTFILE} --output_dir ${OUTPUTDIR} \
--input ${IMG1} --target ${IMG2} --max_timesteps ${ITERS} --alpha ${ALPHA} --lbfgs_max_iterations ${LBFGS} \
--timestepping explicitEuler --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}
# --padding 0 
#--not_store_solver 
# --not_store_solver 
# --solver cg 
########## --preconditioning none 
########## ADD --exclusive IF RUNNING WITHOUT PRECONDITIONING
#########
# --nosmoothen
# #--preconditioning none 
# --starting_guess /home/bastian/D1/registration/affine-rotated-outputs3/E70A0.01LBFGS70prec_class/CurrentControl.hdf --readname "function" \
# --starting_state /home/bastian/D1/registration/affine-rotated-outputs3/E70A0.01LBFGS70prec_class/State_checkpoint.xdmf \


# --tukey --tukey_c 2 
# --tukey --tukey_c 1 
#
# --tukey --tukey_c 1 \
# --tukey \
# --starting_guess /home/bastian/D1/registration/croppedmriregistration_outputs/E100A0.01LBFGS100/VelocityField.hdf --readname "-1" --multigrid \


#
echo "success"
