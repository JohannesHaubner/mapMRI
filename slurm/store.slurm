#!/bin/bash
#SBATCH --job-name=mrireg
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --exclusive
## SBATCH --nodes=3
## SBATCH --ntasks-per-node=32
#################################################################
## SBATCH --partition=milanq
## milanq has 256 cpus per node
#
#SBATCH --partition=fpgaq
## fpgaq has 96 cpus per onde
#
## SBATCH --partition=defq
## has 128 cpus per node
## SBATCH --partition=habanaq
#################################################################
#SBATCH --time 3-00:00:0
#SBATCH -o /global/D1/homes/bastian/registration/storageslurm/%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=bastian@simula.no

eval "$(conda shell.bash hook)"
conda activate mri_inverse
LOGOUTFILE2=/home/bastian/D1/registration/storageslurm/${SLURM_JOB_ID}_log_python_srun.txt


set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

echo "starting to run"

GPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/
SPATH=${GPATH}scripts/optimization
LOGOUTFILE=/home/bastian/D1/registration/storageslurm/${SLURM_JOB_ID}_log_python_srun.txt

OUTPUTDIR=/home/bastian/D1/registration/normalized-states/${SLURM_JOB_ID}/
IMG1=/home/bastian/D1/registration/mri2fem-dataset/normalized/cropped/cropped_abbytoernie_nyul.mgz
IMG2=/home/bastian/D1/registration/mri2fem-dataset/normalized/cropped/cropped_ernie_brain_nyul.mgz
LBFGS=100

OUTPUTDIR=/home/bastian/D1/registration/ventricle-states/${SLURM_JOB_ID}/
IMG1=/home/bastian/D1/registration/hydrocephalus/ventricles/021to068_nyul_ventriclemasked.mgz
IMG2=/home/bastian/D1/registration/hydrocephalus/ventricles/068_brain_nyul_ventriclemasked.mgz
LBFGS=100


NP=1
ITERS=100
ALPHA=1e-2

echo "NP=" ${NP}
echo "ITERS=" ${ITERS}
echo "ALPHA=" ${ALPHA}
echo "OUTPUTDIR=" $OUTPUTDIR
echo "IMG1=" ${IMG1}
echo "IMG2=" ${IMG2}


srun -n ${NP} --cpu-bind=rank python3 -u ${SPATH}/Optimize3d.py \
--logfile ${LOGOUTFILE2} --output_dir ${OUTPUTDIR} \
--input ${IMG1} --target ${IMG2} --max_timesteps ${ITERS} --alpha ${ALPHA} --lbfgs_max_iterations ${LBFGS} \
--storeto "/home/bastian/D1/registration/ventricle-outputs/446036/RKA0.01LBFGS50" \
--huber_delta 1 --timestepping RungeKutta --huber --memdebug --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}


echo "success"