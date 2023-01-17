#!/bin/bash
#SBATCH --job-name=transport
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
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

# FOLDER=croppedmriregistration_outputs
# JOBNAME=OCDA10.0LBFGS5
# VFILE=opts/opt_phi_6.h5
# READNAME="function"

# FOLDER=mriregistration_outputs
# JOBNAME=OCDA0.001LBFGS1000
# VFILE=opts/opt_phi_204.h5
# READNAME="function"

# FOLDER=mriregistration_outputs
# JOBNAME=OCDA0.0001LBFGS100RS1e-4
# VFILE=VelocityField.hdf
# READNAME="-1"



# FOLDER=mriregistration_outputs
# JOBNAME=E100A0.001LBFGS100
# VFILE=Control.hdf
# READNAME="-1"

FOLDER=croppedmriregistration_outputs
JOBNAME=E100A0.01LBFGS100
VFILE=CurrentV.hdf
READNAME="function"



# FOLDER=mriregistration_outputs
# # JOBNAME=E100A0.0001LBFGS100NOSMOOTHEN
# JOBNAME=E100A0.01LBFGS100NOSMOOTHEN
# # JOBNAME=E100A0.0001LBFGS100
# VFILE=VelocityField.hdf
# READNAME="-1"


srun -n 32 python3 -u transform_mesh.py --mapping_only --folder ${FOLDER} --job ${JOBNAME} \
--postfolder postprocessing --velocityfilename ${VFILE} --readname ${READNAME} > ${LOGOUTFILE}
echo "Computed mapping, now running transform with only one core"

srun -n 1 python3 -u transform_mesh.py --folder ${FOLDER} --job ${JOBNAME} \
--postfolder postprocessing --velocityfilename ${VFILE} --readname ${READNAME} > ${LOGOUTFILE}
echo "success"






# 
#   OBSOLETE SETTINGS FOR DIFFERENT HYPERPARAMETERS BELOW

# srun -n 32 python3 -u transform.py --mapping_only --folder ${FOLDER} --job ${JOBNAME} \
# --postfolder processing_dg_inv --DGtransport --inverseRAS \
# --velocityfilename ${VFILE} --readname ${READNAME} > ${LOGOUTFILE}
# echo "Computed mapping, now running transform with only one core"

# srun -n 1 python3 -u transform.py --folder ${FOLDER} --job ${JOBNAME} \
# --postfolder processing_dg_inv --DGtransport --inverseRAS \
# --velocityfilename ${VFILE} --readname ${READNAME} >> ${LOGOUTFILE}
# echo "success"

# srun -n 32 python3 -u transform.py --mapping_only --folder ${FOLDER} --job ${JOBNAME} \
# --postfolder processing_dg --DGtransport \
# --velocityfilename ${VFILE} --readname ${READNAME} > ${LOGOUTFILE}
# echo "Computed mapping, now running transform with only one core"

# srun -n 1 python3 -u transform.py --folder ${FOLDER} --job ${JOBNAME} \
# --postfolder processing_dg --DGtransport \
# --velocityfilename ${VFILE} --readname ${READNAME} >> ${LOGOUTFILE}
# echo "success"

# srun -n 32 python3 -u transform.py --mapping_only --folder ${FOLDER} --job ${JOBNAME} \
# --postfolder processing_cg_inv --inverseRAS \
# --velocityfilename ${VFILE} --readname ${READNAME} > ${LOGOUTFILE}
# echo "Computed mapping, now running transform with only one core"

# srun -n 1 python3 -u transform.py --folder ${FOLDER} --job ${JOBNAME} \
# --postfolder processing_cg_inv --inverseRAS \
# --velocityfilename ${VFILE} --readname ${READNAME} >> ${LOGOUTFILE}
# echo "success"

# srun -n 32 python3 -u transform.py --mapping_only --folder ${FOLDER} --job ${JOBNAME} \
# --postfolder processing_cg \
# --velocityfilename ${VFILE} --readname ${READNAME} > ${LOGOUTFILE}
# echo "Computed mapping, now running transform with only one core"

# srun -n 1 python3 -u transform.py --folder ${FOLDER} --job ${JOBNAME} \
# --postfolder processing_cg \
# --velocityfilename ${VFILE} --readname ${READNAME} >> ${LOGOUTFILE}
# echo "success"