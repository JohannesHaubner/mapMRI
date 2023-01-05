#!/bin/bash
#SBATCH --job-name=ocdregister
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#################################################################
##SBATCH --partition=milanq
#SBATCH --partition=fpgaq
## SBATCH --partition=defq
## SBATCH --partition=slowq
## SBATCH --partition=rome16q
#################################################################

#SBATCH --time 6-24 ##--time=99:00:00
#SBATCH -o /home/bastian/Oscar-Image-Registration-via-Transport-Equation/slurm/mrislurm/%j.out
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
SPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/scripts/optimization

LOGOUTFILE=${SPATH}/slurm/mrislurm/${SLURM_JOB_ID}_log_python_srun.txt


ITERS=99999
ALPHA=$1
LBFGS=99999

echo "ITERS=" ${ITERS}
echo "ALPHA=" ${ALPHA}

OUTFOLDERNAME=OCDA${ALPHA}



OUTPUTDIR=/home/bastian/D1/registration/mriregistration_outputs/
DATAPATH=/home/bastian/D1/registration/mri2fem-data/processed/coarsecropped/

echo "OUTPUTDIR=" $OUTPUTDIR
echo "OUTFOLDERNAME=" $OUTFOLDERNAME

IMG1=${DATAPATH}coarsenedabbytoernie.mgz
IMG2=${DATAPATH}coarsenedernie_brain.mgz

echo "IMG1=" ${IMG1}
echo "IMG2=" ${IMG2}

srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
--input ${IMG1} --target ${IMG2} --ocd \
--max_timesteps ${ITERS} --alpha ${ALPHA} \
--lbfgs_max_iterations ${LBFGS} --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}
#--state_functiondegree 0  --nosmoothen 
#  --timestepping explicitEuler --solver lu

OUTPATH=${OUTPUTDIR}/${OUTFOLDERNAME}/
cp -v ${LOGOUTFILE} ${OUTPATH}

echo "success"