#!/bin/bash
#SBATCH --job-name=register
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#################################################################
##SBATCH --partition=milanq
##SBATCH --partition=fpgaq
#SBATCH --partition=defq
## SBATCH --partition=slowq
## SBATCH --partition=rome16q
#################################################################

#SBATCH --time 6-24 ##--time=99:00:00
#SBATCH -o /home/bastian/Oscar-Image-Registration-via-Transport-Equation/2dslurm/%j.out
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
SPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation
LOGOUTFILE=${SPATH}/2dslurm/${SLURM_JOB_ID}_log_python_srun.txt

# OUTPUTDIR=/home/bastian/D1/mriregistration_outputs/




ITERS=$1
ALPHA=$2
LBFGS=100

OUTFOLDERNAME=RK${ITERS}A${ALPHA}LBFGS${LBFGS}
#PIC2FEN

OUTPUTDIR=/home/bastian/D1/registration/boxregistration_outputs/
DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_2d

# OUTPUTDIR=/home/bastian/D1/registration/cuberegistration_outputs/
# DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_3d

srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
--input ${DATAPATH}/input.mgz --target ${DATAPATH}/target.mgz \
--max_timesteps ${ITERS} --alpha ${ALPHA} --timestepping RungeKutta \
--lbfgs_max_iterations ${LBFGS} --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}
# --Pic2FEN

# OUTFOLDERNAME=RK${ITERS}A${ALPHA}LBFGS${LBFGS}
# OUTPUTDIR=/home/bastian/D1/imageregistration_outputs/

# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --max_timesteps ${ITERS} --alpha ${ALPHA} --timestepping RungeKutta \
# --lbfgs_max_iterations ${LBFGS} --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}


OUTPATH=${OUTPUTDIR}/${OUTFOLDERNAME}/
cp -v ${LOGOUTFILE} ${OUTPATH}



echo "success"