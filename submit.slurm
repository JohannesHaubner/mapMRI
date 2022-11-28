#!/bin/bash
#SBATCH --job-name=register
#SBATCH --cpus-per-task=1 --nodes=4
# #SBATCH --partition=milanq
#SBATCH --partition=fpgaq
## SBATCH --partition=defq
#defq also possible

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


# OUTFOLDERNAME=createStartingGuess
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 100 --solver krylov > ${LOGOUTFILE}




# OUTFOLDERNAME=timetestKrylov
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 10 --solver krylov > ${LOGOUTFILE}

# OUTFOLDERNAME=testRungeKutta
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 50 --solver krylov > ${LOGOUTFILE}



# OUTPUTDIR=/home/bastian/D1/debug_imageregistration/
# mkdir -pv ${OUTPUTDIR}

# OUTFOLDERNAME=RungeKuttaNoSmoothen
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --nosmoothen --solver lu --debug > ${LOGOUTFILE}

# OUTFOLDERNAME=timetestLu
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 10 --solver lu > ${LOGOUTFILE}


# OUTFOLDERNAME=EulerNoSmoothen
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --nosmoothen --solver lu --debug --timestepping explicitEuler > ${LOGOUTFILE}

# OUTFOLDERNAME=CrankNicolsonNoSmoothen
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --nosmoothen --solver lu --debug --timestepping CrankNicolson > ${LOGOUTFILE}


# OUTFOLDERNAME=RungeKuttaFix
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --solver lu --debug > ${LOGOUTFILE}


# OUTFOLDERNAME=EulerFixNoSmoothen
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --solver lu --debug --nosmoothen --timestepping explicitEuler > ${LOGOUTFILE}


# OUTFOLDERNAME=EulerFix
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --solver lu --debug --timestepping explicitEuler > ${LOGOUTFILE}

# OUTFOLDERNAME=CrankNicolsonFix
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --solver lu --debug --timestepping CrankNicolson > ${LOGOUTFILE}


# OUTFOLDERNAME=timetestRungeKutta
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 10 --solver krylov > ${LOGOUTFILE}


# OUTFOLDERNAME=timetestEuler
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 10 --solver krylov --timestepping explicitEuler > ${LOGOUTFILE}

# OUTFOLDERNAME=test
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} \
# --lbfgs_max_iterations 1 --smoothen --solver krylov > ${LOGOUTFILE}


OUTPATH=${OUTPUTDIR}${OUTFOLDERNAME}/
cp -v ${LOGOUTFILE} ${OUTPATH}



echo "success"