#!/bin/bash
#SBATCH --job-name=ventreg
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=256
## SBATCH --exclusive
#SBATCH --nodes=1
## SBATCH --ntasks-per-node=32
#################################################################
#SBATCH --partition=milanq
## milanq has 256 cpus per node
#
## SBATCH --partition=fpgaq
## fpgaq has 96 cpus per onde
#
## SBATCH --partition=defq
## has 128 cpus per node
## SBATCH --partition=habanaq
#################################################################
#SBATCH --time 3-00:00:0
#SBATCH -o /global/D1/homes/bastian/registration/mrislurm/%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=bastian@simula.no

eval "$(conda shell.bash hook)"
conda activate mri_inverse
LOGOUTFILE2=/home/bastian/D1/registration/mrislurm/${SLURM_JOB_ID}_log_python_srun.txt


set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

echo "starting to run"

# module purge
# module load singularity-ce
# module load mpich-3.3.2

# export SINGULARITYENV_CXX=/usr/bin/c++
# export SINGULARITY_DOCKER_USERNAME=johanneshaubner
# export SINGULARITY_DOCKER_PASSWORD=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
# export SINGULARITY_BIND="/global/D1/homes/bastian:/home/bastian/d1"
# OUTPUTDIR=/home/bastian/d1/registration/normalized-outputs/${SLURM_JOB_ID}/
# IMG1=/home/bastian/d1/registration/mri2fem-dataset/normalized/cropped/cropped_abbytoernie_nyul.mgz
# IMG2=/home/bastian/d1/registration/mri2fem-dataset/normalized/cropped/cropped_ernie_brain_nyul.mgz

# # OUTPUTDIR=/home/bastian/d1/registration/normalized-outputs/${SLURM_JOB_ID}/
# # IMG1=/home/bastian/d1/registration/mri2fem-dataset/normalized/cropped/cropped_abbytoernie_nyul.mgz
# # IMG2=/home/bastian/d1/registration/mri2fem-dataset/normalized/cropped/cropped_ernie_brain_nyul.mgz
# LOGOUTFILE2=/home/bastian/d1/registration/mrislurm/${SLURM_JOB_ID}_log_python_srun.txt



#######################################################
## EXECUTE AFTER ANOTHER JOB IS DONE:
## sbatch --dependency=afterany: submit.slurm
#######################################################

GPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/
SPATH=${GPATH}scripts/optimization
LOGOUTFILE=/home/bastian/D1/registration/mrislurm/${SLURM_JOB_ID}_log_python_srun.txt



OUTPUTDIR=/home/bastian/D1/registration/ventricle-outputs/${SLURM_JOB_ID}/
IMG1=/home/bastian/D1/registration/hydrocephalus/ventricles/021to068_nyul_ventriclemasked.mgz
IMG2=/home/bastian/D1/registration/hydrocephalus/ventricles/068_brain_nyul_ventriclemasked.mgz
LBFGS=100

NP=64

echo "NP=" ${NP}

ITERS=100
ALPHA=1e-2

echo "ITERS=" ${ITERS}
echo "ALPHA=" ${ALPHA}

echo "OUTPUTDIR=" $OUTPUTDIR
echo "IMG1=" ${IMG1}
echo "IMG2=" ${IMG2}

# cp -vr /global/D1/homes/bastian/registration/test/445806 /global/D1/homes/bastian/registration/normalized-outputs/

srun -n ${NP} --cpu-bind=rank python3 -u ${SPATH}/Optimize3d.py \
--logfile ${LOGOUTFILE2} --output_dir ${OUTPUTDIR} \
--input ${IMG1} --target ${IMG2} --max_timesteps ${ITERS} --alpha ${ALPHA} --lbfgs_max_iterations ${LBFGS} \
--starting_state /global/D1/homes/bastian/registration/ventricle-outputs/446036/RKA0.01LBFGS50/State_checkpoint.xdmf --readname CurrentState \
--huber_delta 1 --timestepping RungeKutta --huber --memdebug --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}

# 446034
# --starting_state /global/D1/homes/bastian/registration/ventricle-outputs/446034/RKA0.01LBFGS50/State_checkpoint.xdmf --readname CurrentState \

# /global/D1/homes/bastian/registration/ventricle-outputs/445829/RKA0.01LBFGS50/
# --starting_state /home/bastian/D1/registration/ventricle-outputs/445351/RKA0.01LBFGS50/State_checkpoint.xdmf --readname CurrentState \
# --starting_guess /home/bastian/D1/registration/ventricle-outputs/445351/RKA0.01LBFGS50/Control_checkpoint.xdmf --readname CurrentV \

echo "success"
