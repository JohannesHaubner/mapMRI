#!/bin/bash
#SBATCH --job-name=register
#SBATCH --cpus-per-task=1 --nodes=4
## SBATCH --partition=milanq
#SBATCH --partition=fpgaq
## SBATCH --partition=defq
## defq also possible

#SBATCH --time 5-24 ##--time=99:00:00
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

# OUTFOLDERNAME=interpolateTest #createStartingGuess
# startingGuess=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/outputs/exportVtest/Control.hdf
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --timestepping explicitEuler \
# --input mridata_3d/091registeredto205_padded.mgz --target mridata_3d/205_cropped_padded.mgz --interpolate \
# --readname 0 --starting_guess ${startingGuess} --solver ${solver} > ${outfolder}_log_python_srun.txt 

# OUTFOLDERNAME=createStartingGuess
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --timestepping explicitEuler \
# --solver ${solver} --create_guess_only > ${outfolder}_log_python_srun.txt 

# OUTFOLDERNAME=createStartingGuessOnly
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --timestepping explicitEuler \
# --solver ${solver} > ${outfolder}_log_python_srun.txt 

# OUTFOLDERNAME=createStartingGuessOneIter
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --timestepping explicitEuler \
# --solver ${solver} --lbfgs_max_iterations 1 > ${outfolder}_log_python_srun.txt 
 
echo "success"