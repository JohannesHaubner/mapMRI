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
#SBATCH -o /home/bastian/Oscar-Image-Registration-via-Transport-Equation/slurm/cubedebugging/%j.out
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

LOGOUTFILE=${SPATH}/slurm/cubedebugging/${SLURM_JOB_ID}_log_python_srun.txt
OUTPUTDIR=/home/bastian/D1/mriregistration_outputs/
DATAPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/mridata_3d

OUTPUTDIR=/home/bastian/D1/cubedebugging/

OUTFOLDERNAME=$1$2$3$4$5$6DG0
OUTPATH=${OUTPUTDIR}${OUTFOLDERNAME}/
mkdir -pv ${OUTPATH}
cd ${OUTPATH}
mkdir -pv ${OUTPATH}/output
echo "DEBUGGING"
# parser.add_argument("--newTransport",type=str, choices=["true", "false"])
# parser.add_argument("--alpha", default=1e-4, type=float)
# parser.add_argument("--regularize", choices=["L2control", "velocity"])
# parser.add_argument("--loading", choices=["Pic2FEN", "MRI2FEM"])
# parser.add_argument("--normalize", type=str, choices=["true", "false"])
python -u /home/bastian/Oscar-Image-Registration-via-Transport-Equation/Optimize.py --maxiter 800 --maxlbfs 1000 \
--newTransport $1 --alpha $2 --regularize $3 --loading $4 --normalize $5 --cube true --timestepping $6 > ${LOGOUTFILE}




cp -v ${LOGOUTFILE} ${OUTPATH}



echo "success"