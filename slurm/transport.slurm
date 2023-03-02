#!/bin/bash
#SBATCH --job-name=transport
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
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
# srun -n 32 python3 -u transform_mesh.py --mapping_only --folders ${FOLDER} > ${LOGOUTFILE}
# echo "Computed mapping, now running transform with only one core"


DPATH=/home/bastian/D1/registration/ventricle-outputs

srun -n 16 python3 -u transform_mesh.py --mapping_only \
--folders \
${DPATH}/446036/RKA0.01LBFGS50/ \
${DPATH}/446180/RKA0.01LBFGS100/ \
${DPATH}/446431/RK100A0.01LBFGS100/ \
> ${LOGOUTFILE}

echo "Computed mapping, now running transform with only one core"










# ###
# ## TODO !!!!!!!!!!!!!!!
# ## TODO !!!!!!!!!!!!!!!
# ### TODO !!!!!!!!!!!!!!!
# ## change mesh back to 16 resolution
# ## TODO !!!!!!!!!!!!!!!
# ## TODO !!!!!!!!!!!!!!!
# ### TODO !!!!!!!!!!!!!!!
# ## TODO !!!!!!!!!!!!!!!
# ## TODO !!!!!!!!!!!!!!!
# ### TODO !!!!!!!!!!!!!!!

# srun -n 1 python3 -u transform_mesh.py --folders ${FOLDER} \
# --meshoutputfolder /home/bastian/D1/registration/transported-meshes/${SLURM_JOB_ID}/ >> ${LOGOUTFILE}
# echo "success"


