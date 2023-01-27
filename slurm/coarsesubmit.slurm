#!/bin/bash
#SBATCH --job-name=quicktest
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#################################################################
## SBATCH --partition=milanq
## SBATCH --partition=fpgaq
#SBATCH --partition=defq
#SBATCH --reservation=bastian_39
##SBATCH --partition=habanaq
#################################################################
#SBATCH --time 0-3 ##--time=99:00:00
#SBATCH -o /home/bastian/D1/registration/cubeslurm/%j.out
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
SPATH=${GPATH}scripts/optimization
LOGOUTFILE=/home/bastian/D1/registration/cubeslurm/${SLURM_JOB_ID}_log_python_srun.txt



ITERS=2
ALPHA=1e-2
LBFGS=50

echo "ITERS=" ${ITERS}
echo "ALPHA=" ${ALPHA}

OUTFOLDERNAME=None

NC=$1
prec=none
# prec=preconditioning

OUTFOLDERNAME=NC${NC}_${prec}_noTransport_projector_CG
# _nosmoothen

OUTPUTDIR=/home/bastian/D1/registration/coarsememorytests/
# DPATH=/home/bastian/D1/registration
# DATAPATH=${DPATH}/testdata_3d/
# IMG1=${DATAPATH}input.mgz
# IMG2=${DATAPATH}newtarget.mgz


IMG1=/home/bastian/D1/registration/mri2fem-dataset/processed/coarsecropped/coarsenedabbytoernie.mgz
IMG2=/home/bastian/D1/registration/mri2fem-dataset/processed/coarsecropped/coarsenedernie_brain.mgz

srun -n 1 python3 -u ${SPATH}/Poisson.py --maxcor ${NC} > ${LOGOUTFILE}

# srun -n 32 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --logfile ${LOGOUTFILE} --output_dir ${OUTPUTDIR} \
# --input ${IMG1} --target ${IMG2} --max_timesteps ${ITERS} --alpha ${ALPHA} --lbfgs_max_iterations ${LBFGS} --normalization_scale 1 \
# --memdebug --maxcor ${NC} --preconditioning ${prec} --projector \
# --timestepping explicitEuler --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}
# --tukey --tukey_c 1 \
# --tukey --tukey_c 1 \
# --tukey \
# --projector \
# --nosmoothen \

echo "success"