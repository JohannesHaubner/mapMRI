#!/bin/bash
#SBATCH --job-name=transport
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#################################################################
## SBATCH --partition=milanq
#SBATCH --partition=fpgaq
## SBATCH --partition=defq
## SBATCH --partition=slowq
## SBATCH --partition=rome16q
#################################################################

#SBATCH --time 0-8 ##--time=99:00:00
#SBATCH -o /home/bastian/D1/registration/transportslurm/%j.out
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
SPATH=${GPATH}scripts/postprocessing
LOGOUTFILE=/home/bastian/D1/registration/transportslurm/${SLURM_JOB_ID}_log_python_srun.txt

cd $SPATH

# FOLDER=/home/bastian/D1/registration/normalized-outputs/446152/RKA0.01LBFGS100/
# FOLDER=/home/bastian/D1/registration/normalized-outputs/446600/RK100A0.01LBFGS200/
# FOLDER=/home/bastian/D1/registration/ventricle-outputs/447443/RK100A0.01LBFGS100/
# srun -n 64 python3 -u transform_mesh.py --mapping_only --folders ${FOLDER} --recompute_mapping > ${LOGOUTFILE}
# echo "Computed mapping, now running transform with only one core"

# # /home/bastian/D1/registration/normalized-outputs/446600/RK100A0.01LBFGS200/ \
srun -n 1 python3 -u transform_mesh.py --folders \
/home/bastian/D1/registration/normalized-outputs/446152/RKA0.01LBFGS100/ \
/home/bastian/D1/registration/normalized-outputs/446600/RK100A0.01LBFGS200/ \
--noaffine \
--reverse \
--meshoutputfolder /home/bastian/D1/registration/transported-meshes/abby-ventricles/both-noaffine-reversed/ >> ${LOGOUTFILE}
echo "success"
# --reverse  --noaffine \\




DPATH=/home/bastian/D1/registration/ventricle-outputs

# srun -n 64 python3 -u transform_mesh.py --mapping_only \
# --folders \
# ${DPATH}/446719/RK100A0.01LBFGS100/ \
# --recompute_mapping \
# > ${LOGOUTFILE}

# exit

# ${DPATH}/446036/RKA0.01LBFGS50/ \
# ${DPATH}/446180/RKA0.01LBFGS100/ \
# ${DPATH}/446431/RK100A0.01LBFGS100/ \

# echo "Computed mapping, now running meshtransform with only one core"

# srun -n 1 python3 -u transform_mesh.py \
# --folders \
# ${DPATH}/446036/RKA0.01LBFGS50/ \
# ${DPATH}/446180/RKA0.01LBFGS100/ \
# ${DPATH}/446431/RK100A0.01LBFGS100/ \
# ${DPATH}/446719/RK100A0.01LBFGS100/ \
# --remesh \
# --reverse \
# --meshoutputfolder /home/bastian/D1/registration/transported-meshes/021-ventricle-mymesh-reversed/ >> ${LOGOUTFILE}



#--affineonly \

echo "success"
