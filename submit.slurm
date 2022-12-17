#!/bin/bash
#SBATCH --job-name=register
#SBATCH --cpus-per-task=1
## #### SBATCH --nodes=4
#SBATCH --ntasks=4
#################################################################
## SBATCH --partition=milanq
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
LOGOUTFILE=${SPATH}/outs/${SLURM_JOB_ID}_log_python_srun.txt
OUTPUTDIR=/home/bastian/D1/imageregistration_outputs/

DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/mridata_3d


# OUTFOLDERNAME=testWithStartingGuess
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --starting_guess /home/bastian/D1/imageregistration_outputs/createStartingGuess/Control.hdf \
# --lbfgs_max_iterations 10 --alpha 1e-5 > ${LOGOUTFILE}


# OUTFOLDERNAME=interpolateStartingGuess
# srun -n 4 python3 -u ${SPATH}/mri_utils/Interpolate_Guess.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --fun /home/bastian/D1/imageregistration_outputs/createStartingGuess/Control.hdf > ${LOGOUTFILE}

# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --starting_guess ${OUTPUTDIR}${OUTFOLDERNAME}/Interpolated_Control.hdf --readname refined_control \
# --input ${DATAPATH}/091registeredto205_padded.mgz --target ${DATAPATH}/205_cropped_padded.mgz \
# --lbfgs_max_iterations 1 --alpha 1e-5 > ${LOGOUTFILE}

# OUTFOLDERNAME=createFineStartingGuessA1e-5
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --input ${DATAPATH}/091registeredto205_padded.mgz --target ${DATAPATH}/205_cropped_padded.mgz \
# --lbfgs_max_iterations 50 --alpha 1e-5 > ${LOGOUTFILE}

# OUTFOLDERNAME=createFineStartingGuessA1e-6
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --input ${DATAPATH}/091registeredto205_padded.mgz --target ${DATAPATH}/205_cropped_padded.mgz \
# --lbfgs_max_iterations 50 --alpha 1e-6 > ${LOGOUTFILE}

# OUTFOLDERNAME=createFineStartingGuessA1e-7
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --input ${DATAPATH}/091registeredto205_padded.mgz --target ${DATAPATH}/205_cropped_padded.mgz \
# --lbfgs_max_iterations 50 --alpha 1e-7 > ${LOGOUTFILE}

# --interpolate --debug

# OUTFOLDERNAME=createStartingGuessNoSmoothen
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 100 --nosmoothen > ${LOGOUTFILE}


# OUTFOLDERNAME=smoothen100
# OUTPUTDIR=/home/bastian/D1/testregistration/
# DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_3d
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --input ${DATAPATH}/input.mgz --target ${DATAPATH}/target.mgz \
# --lbfgs_max_iterations 100 > ${LOGOUTFILE}

OUTFOLDERNAME=smoothen100Buffer02
OUTPUTDIR=/home/bastian/D1/testregistration/
DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_3d
srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
--input ${DATAPATH}/input.mgz --target ${DATAPATH}/target.mgz \
--lbfgs_max_iterations 100 --dt_buffer 0.2 > ${LOGOUTFILE}


# OUTFOLDERNAME=nosmoothen100Buffer02
# OUTPUTDIR=/home/bastian/D1/testregistration/
# DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_3d
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --input ${DATAPATH}/input.mgz --target ${DATAPATH}/target.mgz \
# --lbfgs_max_iterations 100 --nosmoothen --dt_buffer 0.2 > ${LOGOUTFILE}


# OUTFOLDERNAME=createStartingGuessNoSmoothenBuffer
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 100 --nosmoothen --dt_buffer 0.2 > ${LOGOUTFILE}

# OUTFOLDERNAME=createStartingGuess
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 100 > ${LOGOUTFILE}

# OUTFOLDERNAME=createStartingGuessBuffer
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 100 --dt_buffer 0.2 > ${LOGOUTFILE}

# OUTFOLDERNAME=createStartingGuessBuffer0.1A1e-5
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 100 --dt_buffer 0.1 --alpha 1e-5 > ${LOGOUTFILE}

# OUTFOLDERNAME=createStartingGuessBufferA1e-6
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 100 --dt_buffer 0.2 --solver lu --alpha 1e-6 > ${LOGOUTFILE}

# OUTFOLDERNAME=createStartingGuessBufferA1e-7
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 100 --dt_buffer 0.2 --solver lu --alpha 1e-7 > ${LOGOUTFILE}

# OUTFOLDERNAME=createStartingGuessBufferA1e-5
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 100 --dt_buffer 0.2 --solver lu --alpha 1e- > ${LOGOUTFILE}


# OUTFOLDERNAME=createStartingGuessBufferA1e-5
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 100 --dt_buffer 0.2 --solver lu --alpha 1e-5 > ${LOGOUTFILE}

# OUTFOLDERNAME=createStartingGuessBuffer0.1
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 100 --dt_buffer 0.1 --solver lu > ${LOGOUTFILE}


# OUTFOLDERNAME=createStartingGuess10
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 10 --dt_buffer 0.2 > ${LOGOUTFILE}

#################
# ARE THESE FINISHED=?
# OUTFOLDERNAME=createStartingGuessA1e-5
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 100 --alpha 1e-5 > ${LOGOUTFILE}

# OUTFOLDERNAME=createStartingGuessA1e-6
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 100 --alpha 1e-6 > ${LOGOUTFILE}

# OUTFOLDERNAME=createStartingGuessA1e-7
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --lbfgs_max_iterations 100 --alpha 1e-7 > ${LOGOUTFILE}



OUTPATH=${OUTPUTDIR}${OUTFOLDERNAME}/
cp -v ${LOGOUTFILE} ${OUTPATH}



echo "success"