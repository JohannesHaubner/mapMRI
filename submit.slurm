#!/bin/bash
#SBATCH --job-name=register
#SBATCH --cpus-per-task=1
## #### SBATCH --nodes=4
#SBATCH --ntasks=4
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


#######################################################
## EXECUTE AFTER ANOTHER JOB IS DONE:
## sbatch --dependency=afterany: submit.slurm
#######################################################
SPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation
LOGOUTFILE=${SPATH}/slurm/${SLURM_JOB_ID}_log_python_srun.txt
OUTPUTDIR=/home/bastian/D1/imageregistration_outputs/
DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/mridata_3d


### Tests

# OUTFOLDERNAME=smoothen1000Buffer005
# OUTPUTDIR=/home/bastian/D1/testregistration/
# DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_3d
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
# --input ${DATAPATH}/input.mgz --target ${DATAPATH}/target.mgz \
# --lbfgs_max_iterations 1000 --dt_buffer 0.05 --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}

OUTFOLDERNAME=nosmoothen1000Buffer005
OUTPUTDIR=/home/bastian/D1/testregistration/
DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_3d
srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
--input ${DATAPATH}/input.mgz --target ${DATAPATH}/target.mgz --nosmoothen \
--lbfgs_max_iterations 1000 --dt_buffer 0.05 --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}


OUTPATH=${OUTPUTDIR}${OUTFOLDERNAME}/
cp -v ${LOGOUTFILE} ${OUTPATH}



echo "success"