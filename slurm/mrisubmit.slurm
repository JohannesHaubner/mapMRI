#!/bin/bash
#SBATCH --job-name=mriregister
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=64
#SBATCH --exclusive
### =user
#SBATCH --ntasks-per-node=64
#################################################################
## SBATCH --partition=milanq
## SBATCH --partition=fpgaq
##
#SBATCH --partition=defq
#SBATCH --reservation=bastian_39
##
##SBATCH --partition=habanaq
#################################################################
#SBATCH --time 13-00:00:0
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



# ITERS=$1
# ALPHA=$2
ITERS=90
ALPHA=1e-2
LBFGS=100

echo "ITERS=" ${ITERS}
echo "ALPHA=" ${ALPHA}

OUTFOLDERNAME=None




OUTPUTDIR=/home/bastian/D1/registration/hopefully_final/
IMG1=/home/bastian/D1/registration/mri2fem-dataset/processed/affine-rotated/abby.mgz
IMG2=/home/bastian/D1/registration/mri2fem-dataset/processed/affine-rotated/ernie.mgz

# OUTPUTDIR=/home/bastian/D1/registration/affine_croppedmriregistration_outputs/tukeyC1/
# IMG1=/home/bastian/D1/registration/mri2fem-dataset/processed/affine_registered/cropped_abbytoernie_affine.mgz
# IMG2=/home/bastian/D1/registration/mri2fem-dataset/processed/affine_registered/cropped_ernie_brain.mgz


mkdir -pv ${OUTPUTDIR}
echo "OUTPUTDIR=" $OUTPUTDIR
echo "OUTFOLDERNAME=" $OUTFOLDERNAME
echo "IMG1=" ${IMG1}
echo "IMG2=" ${IMG2}


srun -n 64 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --logfile ${LOGOUTFILE} --output_dir ${OUTPUTDIR} \
--input ${IMG1} --target ${IMG2} --max_timesteps ${ITERS} --alpha ${ALPHA} --lbfgs_max_iterations ${LBFGS} \
--tukey --tukey_c 1 \
--timestepping explicitEuler --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}
# --tukey --tukey_c 1 \
# --tukey \
# --starting_guess /home/bastian/D1/registration/croppedmriregistration_outputs/E100A0.01LBFGS100/VelocityField.hdf --readname "-1" --multigrid \


#
echo "success"
