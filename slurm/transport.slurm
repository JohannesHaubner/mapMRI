#!/bin/bash
#SBATCH --job-name=transport
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
#################################################################
## SBATCH --partition=milanq
## SBATCH --partition=fpgaq
#SBATCH --partition=fpgaq,milanq
# ,slowq
## SBATCH --partition=defq
## SBATCH --partition=slowq
## SBATCH --partition=rome16q
#################################################################

#SBATCH --time 0-6 ##--time=99:00:00
#SBATCH -o /home/bastian/D1/registration/transportslurm/%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=bastian@simula.no
#module load gcc
#module load anaconda3/x86_64
#export LD_LIBRARY_PATH=/home/bastian/.conda/envs/mri_inverse/lib
# conda init bash

eval "$(conda shell.bash hook)"
conda activate svmtk

# module load ipopt-3.13.3

set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

echo "starting to run"


#######################################################
## EXECUTE AFTER ANOTHER JOB IS DONE:
## sbatch --dependency=afterany: submit.slurm
#######################################################


GPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/
SPATH=${GPATH}scripts/postprocessing
LOGOUTFILE=/home/bastian/D1/registration/transportslurm/${SLURM_JOB_ID}_log_python_srun.txt

cd $SPATH

# OUTPUTFOLDER=cgmeshtransform

# # FOLDER=/home/bastian/D1/registration/normalized-outputs/446152/RKA0.01LBFGS100/
# # # FOLDER=/home/bastian/D1/registration/normalized-outputs/446600/RK100A0.01LBFGS200/
# # # FOLDER=/home/bastian/D1/registration/normalized-outputs/447918/RK100A0.01LBFGS150/
# FOLDER=/home/bastian/D1/registration/normalized-outputs/450276/RK100A0.01LBFGS150/ #alpha=0.5=beta
# # # # not needed: FOLDER=/home/bastian/D1/registration/normalized-outputs/449047/RK100A0.01LBFGS150/ # alpha=0.1, beta=0.9

OUTPUTFOLDER=469194


# New folders:
FOLDER=/home/bastian/D1/registration/smoothened-outputs/${OUTPUTFOLDER}/RK100A0.01LBFGS200/
srun -n 32 python3 -u transform_mesh.py --mapping_only --folders ${FOLDER} --recompute_mapping --outputfoldername ${OUTPUTFOLDER} > ${LOGOUTFILE}
echo "Computed mapping, now running transform with only one core"



# 
python -c "import SVMTK as svmtk"

# /home/bastian/D1/registration/normalized-outputs/446600/RK100A0.01LBFGS200/ \
srun -n 1 python3 -u transform_mesh.py --folders \
/home/bastian/D1/registration/smoothened-outputs/${OUTPUTFOLDER}/RK100A0.01LBFGS200/ \
--noaffine \
--meshoutputfolder /home/bastian/D1/registration/transported-meshes/abby-lh-smoothened/${OUTPUTFOLDER}/ \
--outputfoldername ${OUTPUTFOLDER} >> ${LOGOUTFILE}
echo "success"
#  /home/bastian/D1/registration/normalized-outputs/450276/RK100A0.01LBFGS150/ \
# /home/bastian/D1/registration/normalized-outputs/446152/RKA0.01LBFGS100/ \
# /home/bastian/D1/registration/normalized-outputs/446600/RK100A0.01LBFGS200/ \
# /home/bastian/D1/registration/normalized-outputs/447918/RK100A0.01LBFGS150/ \



DPATH=/home/bastian/D1/registration/ventricle-outputs


echo "success"
