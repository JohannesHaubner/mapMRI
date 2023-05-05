#!/bin/bash
#SBATCH --job-name=mrireg
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=96
#SBATCH --exclusive
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=32
#################################################################
#SBATCH --partition=defq,milanq
#################################################################
#SBATCH --time 6-23:00:0
#SBATCH -o ./outputs/mrislurm/%j.out
#SBATCH --export=ALL

eval "$(conda shell.bash hook)"
conda activate dgregister-env


set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

echo "starting to run"


GPATH=./
SPATH=${GPATH}scripts/3-optimization
LOGOUTFILE=./outputs/mrislurm/${SLURM_JOB_ID}_log_python_srun.txt

OUTPUTDIR=./outputs/smoothened-outputs/${SLURM_JOB_ID}/
IMG1=./data/normalized/cropped/cropped_abbytoernie_nyul.mgz
IMG2=./data/normalized/cropped/cropped_ernie_brain_nyul.mgz

mkdir -pv $OUTPUTDIR

NP=96

echo "NP=" ${NP}
echo "OUTPUTDIR=" $OUTPUTDIR
echo "IMG1=" ${IMG1}
echo "IMG2=" ${IMG2}

srun -n ${NP} python3 -u ${SPATH}/Optimize3d.py \
--logfile ${LOGOUTFILE2} --output_dir ${OUTPUTDIR} \
--input ${IMG1} --target ${IMG2} \
--slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}

echo "success"
