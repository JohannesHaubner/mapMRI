#!/bin/bash
#SBATCH --job-name=register
#SBATCH --cpus-per-task=1
## #### SBATCH --nodes=4
#SBATCH --ntasks=1
#################################################################
##SBATCH --partition=milanq
## SBATCH --partition=fpgaq
#SBATCH --partition=defq
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


#######################################################
## EXECUTE AFTER ANOTHER JOB IS DONE:
## sbatch --dependency=afterany: submit.slurm
#######################################################
SPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation
LOGOUTFILE=${SPATH}/slurm/${SLURM_JOB_ID}_log_python_srun.txt
OUTPUTDIR=/home/bastian/D1/imageregistration_outputs/
DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/mridata_3d


### Tests


# OUTPUTDIR=/home/bastian/D1/hyperparameterstudy_2d/
# DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_2d
# ITERS=$1
# ALPHA=$2




# OUTFOLDERNAME=smoothN${ITERS}A${ALPHA}
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --input ${DATAPATH}/input.mgz --target ${DATAPATH}/target.mgz \
# --max_timesteps ${ITERS} --alpha ${ALPHA} \
# --lbfgs_max_iterations 1000 --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}


# OUTFOLDERNAME=N${ITERS}A${ALPHA}
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --input ${DATAPATH}/input.mgz --target ${DATAPATH}/target.mgz \
# --max_timesteps ${ITERS} --alpha ${ALPHA} \
# --lbfgs_max_iterations 1000 --nosmoothen --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}

OUTPUTDIR=/home/bastian/D1/debugging/

# OUTFOLDERNAME=RocketOriginalTransport
# OUTPATH=${OUTPUTDIR}${OUTFOLDERNAME}/
# mkdir -pv ${OUTPATH}
# cd ${OUTPATH}
# python -u /home/bastian/Oscar-Image-Registration-via-Transport-Equation/Optimize.py > ${LOGOUTFILE}

# OUTFOLDERNAME=CubeOriginalTransport
# OUTPATH=${OUTPUTDIR}${OUTFOLDERNAME}/
# mkdir -pv ${OUTPATH}
# cd ${OUTPATH}
# python -u /home/bastian/Oscar-Image-Registration-via-Transport-Equation/Optimize.py > ${LOGOUTFILE}

# OUTFOLDERNAME=RocketNewTransport
# OUTPATH=${OUTPUTDIR}${OUTFOLDERNAME}/
# mkdir -pv ${OUTPATH}
# cd ${OUTPATH}
# python -u /home/bastian/Oscar-Image-Registration-via-Transport-Equation/Optimize.py > ${LOGOUTFILE}

# OUTFOLDERNAME=CubeNewTransport
# OUTPATH=${OUTPUTDIR}${OUTFOLDERNAME}/
# mkdir -pv ${OUTPATH}
# cd ${OUTPATH}
# python -u /home/bastian/Oscar-Image-Registration-via-Transport-Equation/Optimize.py > ${LOGOUTFILE}


# OUTFOLDERNAME=RocketNewCode
# OUTPATH=${OUTPUTDIR}${OUTFOLDERNAME}/
# mkdir -pv ${OUTPATH}
# cd ${OUTPATH}
# DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation

# srun -n 1 python3 -u /home/bastian/Oscar-Image-Registration-via-Transport-Equation/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --slurmid ${SLURM_JOB_ID} \
# --lbfgs_max_iterations 1000 --max_timesteps 800 --input ${DATAPATH}/shuttle_small.png --target ${DATAPATH}/shuttle_goal.png > ${LOGOUTFILE}

OUTFOLDERNAME=RocketNewCodeRungeKutta
OUTPATH=${OUTPUTDIR}${OUTFOLDERNAME}/
mkdir -pv ${OUTPATH}
cd ${OUTPATH}
DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation

srun -n 1 python3 -u /home/bastian/Oscar-Image-Registration-via-Transport-Equation/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
--slurmid ${SLURM_JOB_ID} --timestepping RungeKutta \
--lbfgs_max_iterations 1000 --max_timesteps 800 --input ${DATAPATH}/shuttle_small.png --target ${DATAPATH}/shuttle_goal.png > ${LOGOUTFILE}


# OUTFOLDERNAME=CubeNewCodeCrankNicolson
# OUTPATH=${OUTPUTDIR}${OUTFOLDERNAME}/
# mkdir -pv ${OUTPATH}
# cd ${OUTPATH}
# DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_2d

# srun -n 1 python3 -u /home/bastian/Oscar-Image-Registration-via-Transport-Equation/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --slurmid ${SLURM_JOB_ID} --timestepping CrankNicolson \
# --lbfgs_max_iterations 1000 --max_timesteps 800 \
# --input ${DATAPATH}/input.mgz --target ${DATAPATH}/target.mgz > ${LOGOUTFILE}

# OUTFOLDERNAME=CubeNewCodeRungeKutta
# OUTPATH=${OUTPUTDIR}${OUTFOLDERNAME}/
# mkdir -pv ${OUTPATH}
# cd ${OUTPATH}
# DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_2d

# srun -n 1 python3 -u /home/bastian/Oscar-Image-Registration-via-Transport-Equation/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --slurmid ${SLURM_JOB_ID} --timestepping RungeKutta \
# --lbfgs_max_iterations 1000 --max_timesteps 800 \
# --input ${DATAPATH}/input.mgz --target ${DATAPATH}/target.mgz > ${LOGOUTFILE}



cp -v ${LOGOUTFILE} ${OUTPATH}



echo "success"