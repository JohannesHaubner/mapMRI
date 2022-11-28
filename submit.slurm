#!/bin/bash
#SBATCH --job-name=register
#SBATCH --cpus-per-task=1 --nodes=4
#################################################################
## SBATCH --partition=milanq
#SBATCH --partition=fpgaq
## SBATCH --partition=defq
## SBATCH --partition=slowq
## SBATCH --partition=rome16q
#################################################################

#SBATCH --time 6-24 ##--time=99:00:00
#SBATCH -o /home/bastian/Oscar-Image-Registration-via-Transport-Equation/slurm/%j.out
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

SPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation
LOGOUTFILE=${SPATH}/slurm/${SLURM_JOB_ID}_log_python_srun.txt
OUTPUTDIR=/home/bastian/D1/imageregistration_outputs/



# OUTFOLDERNAME=createStartingGuessNoSmoothenRome16
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 10 --nosmoothen > ${LOGOUTFILE}


# OUTFOLDERNAME=perturbEuler
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 10 --timestepping explicitEuler --vinit 0.1 > ${LOGOUTFILE}


# OUTFOLDERNAME=perturbCn
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 10 --timestepping CrankNicolson --vinit 0.1 > ${LOGOUTFILE}



# OUTFOLDERNAME=perturbVinit
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 10 --vinit 0.1 > ${LOGOUTFILE}

OUTFOLDERNAME=noperturb
srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
--lbfgs_max_iterations 10 > ${LOGOUTFILE}


# OUTFOLDERNAME=createStartingGuess
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 10 --solver krylov > ${LOGOUTFILE}

# OUTFOLDERNAME=timeTestLu
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 10 --solver lu > ${LOGOUTFILE}



# OUTFOLDERNAME=createStartingGuessEuler
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 10 --timestepping explicitEuler --solver krylov > ${LOGOUTFILE}


OUTPATH=${OUTPUTDIR}${OUTFOLDERNAME}/
cp -v ${LOGOUTFILE} ${OUTPATH}



echo "success"