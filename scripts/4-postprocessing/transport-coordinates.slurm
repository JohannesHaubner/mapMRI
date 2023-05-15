#!/bin/bash
#SBATCH --job-name=transport
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
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

LOGOUTFILE=./outputs/mrislurm/${SLURM_JOB_ID}_log_python_srun.txt

FOLDER=./outputs/my_registration_1/

srun -n 32 python3 -u ./scripts/4-postprocessing/transform_coordinates.py \
--folder ${FOLDER} \
--outputfilename testall.hdf > ${LOGOUTFILE}

echo "success"
