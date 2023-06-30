#!/bin/bash
#SBATCH --job-name=transport
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=64
#################################################################
#SBATCH --partition=fpgaq
#################################################################
#SBATCH --time 0-6 ##--time=99:00:00
#SBATCH --export=ALL
#SBATCH -o ./outputs/mrislurm/%j.out


eval "$(conda shell.bash hook)"
conda activate dgregister-env

set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

# Check if SVMTK can be imported
python -c "import SVMTK as svmtk"

echo "starting to run"

NP=64

LOGOUTFILE=./outputs/mrislurm/${SLURM_JOB_ID}_log_python_srun.txt

GPATH=./
SPATH=${GPATH}scripts/3-optimization
IMG1=./data/normalized/cropped/cropped_abby2ernie_norm.mgz
IMG2=${IMG1}

ALPHA=0
BETA=1


FOLDER1=./outputs/my_registration_1/
OUTPUTDIR=${FOLDER1}norm_registered/
srun -n ${NP} python3 -u ${SPATH}/Optimize3d.py --output_dir ${OUTPUTDIR} \
--alpha $ALPHA --beta $BETA --forward --starting_guess ${FOLDER1}Control_checkpoint.xdmf \
--input ${IMG1} --target ${IMG2} --max_timesteps 100 > ${LOGOUTFILE}

FOLDER1=./outputs/my_registration_2/
PREVIMAGE=/home/bastian/mapMRI/outputs/my_registration_1/norm_registered/Finalstate.xdmf
OUTPUTDIR=${FOLDER1}norm_registered/
srun -n ${NP} python3 -u ${SPATH}/Optimize3d.py --output_dir ${OUTPUTDIR} \
--alpha $ALPHA --beta $BETA --forward --starting_guess ${FOLDER1}Control_checkpoint.xdmf \
--starting_state ${PREVIMAGE} --statename Finalstate \
--input ${IMG1} --target ${IMG2} --max_timesteps 100 > ${LOGOUTFILE}


FOLDER1=./outputs/my_registration_3/
PREVIMAGE=/home/bastian/mapMRI/outputs/my_registration_2/norm_registered/Finalstate.xdmf
OUTPUTDIR=${FOLDER1}norm_registered/
srun -n ${NP} python3 -u ${SPATH}/Optimize3d.py --output_dir ${OUTPUTDIR} \
--alpha $ALPHA --beta $BETA --forward --starting_guess ${FOLDER1}Control_checkpoint.xdmf \
--starting_state ${PREVIMAGE} --statename Finalstate \
--input ${IMG1} --target ${IMG2} --max_timesteps 100 > ${LOGOUTFILE}


FOLDER1=./outputs/my_registration_4/
PREVIMAGE=/home/bastian/mapMRI/outputs/my_registration_3/norm_registered/Finalstate.xdmf
ALPHA=0.5
BETA=0.5
OUTPUTDIR=${FOLDER1}norm_registered/

srun -n ${NP} python3 -u ${SPATH}/Optimize3d.py --output_dir ${OUTPUTDIR} \
--alpha $ALPHA --beta $BETA --forward --starting_guess ${FOLDER1}Control_checkpoint.xdmf \
--starting_state ${PREVIMAGE} --statename Finalstate \
--input ${IMG1} --target ${IMG2} --max_timesteps 100 > ${LOGOUTFILE}


echo "success"
