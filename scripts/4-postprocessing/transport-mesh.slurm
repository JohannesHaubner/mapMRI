#!/bin/bash
#SBATCH --job-name=transport
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
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

FOLDER1=./outputs/my_registration_1/
FOLDER2=./outputs/my_registration_2/
FOLDER3=./outputs/my_registration_3/
FOLDER4=./outputs/my_registration_4/

MESHFILE=./data/meshes/abby/lh_affine_registered/lh.xml

OUTFOLDER=./outputs/meshregistration_test-4/


srun -n 1 python3 -u ./scripts/4-postprocessing/transform_mesh.py --remesh \
--folders ${FOLDER1} ${FOLDER2} ${FOLDER3} ${FOLDER4} \
--input_meshfile ${MESHFILE} --meshoutputfolder ${OUTFOLDER} > ${LOGOUTFILE}
echo "success"
