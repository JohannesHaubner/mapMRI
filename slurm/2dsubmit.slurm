#!/bin/bash
#SBATCH --job-name=testregister
#SBATCH --cpus-per-task=1
#################################################################
#SBATCH --ntasks=8
################################################################
## SBATCH --partition=milanq
#SBATCH --partition=fpgaq
## SBATCH --partition=defq
## SBATCH --partition=slowq
## SBATCH --partition=rome16q
#################################################################

#SBATCH --time 6-24 ##--time=99:00:00
#SBATCH -o /home/bastian/Oscar-Image-Registration-via-Transport-Equation/slurm/2dslurm/%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=bastian@simula.no

eval "$(conda shell.bash hook)"
conda activate mri_inverse

set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

echo "starting to run"


#######################################################
## EXECUTE AFTER ANOTHER JOB IS DONE:
## sbatch --dependency=afterany: submit.slurm
#######################################################
SPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation
LOGOUTFILE=${SPATH}/2dslurm/${SLURM_JOB_ID}_log_python_srun.txt



ITERS=$1
ALPHA=$2

NP=8
LBFGS=20
OUTFOLDERNAME=${NP}
OUTPUTDIR=/home/bastian/D1/registration/timing3d/
DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_3d

# LBFGS=10000
# OUTFOLDERNAME=None
# OUTPUTDIR=/home/bastian/D1/registration/boxregistration_outputs/
# DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_2d

# OUTFOLDERNAME=None
# OUTPUTDIR=/home/bastian/D1/registration/cuberegistration_outputs/
# DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_3d

srun -n ${NP} python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --logfile ${LOGOUTFILE} --output_dir ${OUTPUTDIR} \
--input ${DATAPATH}/input.mgz --target ${DATAPATH}/target.mgz \
--max_timesteps ${ITERS} --alpha ${ALPHA} \
--lbfgs_max_iterations ${LBFGS} --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}
#--state_functiondegree 0  --nosmoothen --timestepping explicitEuler

# # --timestepping explicitEuler
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --logfile ${LOGOUTFILE} --output_dir ${OUTPUTDIR} \
# --input ${DATAPATH}/input.mgz --target ${DATAPATH}/target.mgz \
# --max_timesteps ${ITERS} --alpha ${ALPHA} --timestepping explicitEuler --nosmoothen \
# --lbfgs_max_iterations ${LBFGS} --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}

# # --nosmoothen
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --logfile ${LOGOUTFILE} --output_dir ${OUTPUTDIR} \
# --input ${DATAPATH}/input.mgz --target ${DATAPATH}/target.mgz \
# --max_timesteps ${ITERS} --alpha ${ALPHA} --nosmoothen \
# --lbfgs_max_iterations ${LBFGS} --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}

# # --state_functiondegree 0 
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --logfile ${LOGOUTFILE} --output_dir ${OUTPUTDIR} \
# --input ${DATAPATH}/input.mgz --target ${DATAPATH}/target.mgz \
# --max_timesteps ${ITERS} --alpha ${ALPHA} --state_functiondegree 0 \
# --lbfgs_max_iterations ${LBFGS} --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}




# OUTPATH=${OUTPUTDIR}/${OUTFOLDERNAME}/
# cp -v ${LOGOUTFILE} ${OUTPUTDIR}/${OUTFOLDERNAME}/

echo "success"