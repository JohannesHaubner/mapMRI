#!/bin/bash
#SBATCH --job-name=register
#SBATCH --cpus-per-task=1 --nodes=4
#SBATCH --partition=milanq
## SBATCH --partition=fpgaq
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

SPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation



set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

echo "starting to run"

outfolder=${SPATH}/slurm/${SLURM_JOB_ID}
solver=lu


# OUTFOLDERNAME=interpolateTest4_sequential
# startingGuess=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/outputs/readStartingGuess/Control.hdf
# srun -n 1 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} \
# --input mridata_3d/091registeredto205_padded.mgz --target mridata_3d/205_cropped_padded.mgz --interpolate \
# --readname 400 --starting_guess ${startingGuess} --solver ${solver} > ${outfolder}_log_python_srun.txt 
# # # run in parallel
# # startingGuess=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/outputs/interpolateTest/Interpolated_Control.hdf
# # srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} \
# # --input mridata_3d/091registeredto205_padded.mgz --target mridata_3d/205_cropped_padded.mgz \
# # --readname control --starting_guess ${startingGuess} --solver ${solver} > ${outfolder}_log_python_srun.txt 



# OUTFOLDERNAME=fullResolutionRegister
# startingGuess=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/outputs/createTestStartingGuess/Control.hdf
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} \
# --input mridata_3d/091registeredto205_padded.mgz --target mridata_3d/205_cropped_padded.mgz \
# --solver ${solver} > ${outfolder}_log_python_srun.txt 


# OUTFOLDERNAME=rungeKuttaTest
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} \
# --solver ${solver} > ${outfolder}_log_python_srun.txt 



# OUTFOLDERNAME=outputTestWStartingGuess
# startingGuess=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/outputs/readStartingGuess/Control.hdf
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} \
# --readname 400 --alpha 1e-5 --lbfgs_max_iterations 2 --starting_guess ${startingGuess} --solver ${solver} > ${outfolder}_log_python_srun.txt

# OUTFOLDERNAME=outputTest
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --lbfgs_max_iterations 2 \
# --alpha 1e-5 --solver ${solver} --lbfgs_max_iterations 2 > ${outfolder}_log_python_srun.txt

# OUTFOLDERNAME=readStartingGuess
# startingGuess=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/outputs/createTestStartingGuess/Control.hdf
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} \
# --readname 1 --starting_guess ${startingGuess} --solver ${solver} > ${outfolder}_log_python_srun.txt


OUTFOLDERNAME=createTestStartingGuessEuler
srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} \
--solver ${solver} --timestepping explicitEuler --lbfgs_max_iterations 40 > ${outfolder}_log_python_srun.txt 


# OUTFOLDERNAME=createTestStartingGuess
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} \
# --solver ${solver} --lbfgs_max_iterations 2 > ${outfolder}_log_python_srun.txt 
 
OUTPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/outputs/${OUTFOLDERNAME}/
cp -v /home/bastian/Oscar-Image-Registration-via-Transport-Equation/slurm/${SLURM_JOB_ID}_log_python_srun.txt ${OUTPATH}

echo "success"