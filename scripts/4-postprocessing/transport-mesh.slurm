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
conda activate svmtk

set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

# Check if SVMTK can be imported
python -c "import SVMTK as svmtk"


echo "starting to run"


#######################################################
## EXECUTE AFTER ANOTHER JOB IS DONE:
## sbatch --dependency=afterany: submit.slurm
#######################################################


GPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/
SPATH=${GPATH}scripts/postprocessing
LOGOUTFILE=/home/bastian/D1/registration/transportslurm/${SLURM_JOB_ID}_log_python_srun.txt

cd $SPATH

OUTPUTFOLDER=469194


# New folders:
FOLDER=/home/bastian/D1/registration/smoothened-outputs/${OUTPUTFOLDER}/RK100A0.01LBFGS200/
srun -n 32 python3 -u transform_mesh.py --mapping_only --folders ${FOLDER} --recompute_mapping --outputfoldername ${OUTPUTFOLDER} > ${LOGOUTFILE}
echo "Computed mapping, now running transform with only one core"


srun -n 1 python3 -u transform_mesh.py --folders \
/home/bastian/D1/registration/smoothened-outputs/${OUTPUTFOLDER}/RK100A0.01LBFGS200/ \
--noaffine \
--meshoutputfolder /home/bastian/D1/registration/transported-meshes/abby-lh-smoothened/${OUTPUTFOLDER}/ \
--outputfoldername ${OUTPUTFOLDER} >> ${LOGOUTFILE}

echo "success"
