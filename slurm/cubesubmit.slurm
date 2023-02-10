#!/bin/bash
#SBATCH --job-name=quicktest
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=32
#SBATCH --nodes=1
#################################################################
## SBATCH --partition=milanq
#SBATCH --partition=fpgaq
## SBATCH --partition=defq
#################################################################
##SBATCH --time 1-00:00:0
#SBATCH --time 0-24 ##--time=99:00:00
#SBATCH -o /home/bastian/D1/registration/cubeslurm/%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=bastian@simula.no

eval "$(conda shell.bash hook)"
conda activate mri_inverse
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


NP=32

ITERS=32
ALPHA=1e-6
LBFGS=30

echo "ITERS=" ${ITERS}
echo "ALPHA=" ${ALPHA}

echo "NTASKS=" ${NP}

OUTFOLDERNAME=None
OUTFOLDERNAME=nosmoothen

OUTPUTDIR=/home/bastian/D1/registration/cuberegistration/
DPATH=/home/bastian/D1/registration
DATAPATH=${DPATH}/testdata_3d/
IMG1=${DATAPATH}input.mgz
IMG2=${DATAPATH}newtarget.mgz

mkdir -pv ${OUTPUTDIR}

# GUESSPATH=/home/bastian/D1/registration/cubememdebug2/441137/E32A0.1LBFGS30nosmoothen
GUESSPATH=/home/bastian/D1/registration/cuberegistration/E32A0.0001LBFGS30nosmoothen

srun -n ${NP} python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --logfile ${LOGOUTFILE} --output_dir ${OUTPUTDIR} \
--input ${IMG1} --target ${IMG2} --max_timesteps ${ITERS} --alpha ${ALPHA} --lbfgs_max_iterations ${LBFGS} --normalization_scale 1 \
--starting_state ${GUESSPATH}/State_checkpoint.xdmf \
--timestepping explicitEuler --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}

# --nosmoothen  --preconditioning none \
# RungeKutta
# --tukey --tukey_c 1 \
# --tukey --tukey_c 1 \
# --tukey \

scp ${LOGOUTFILE} ${OUTPUTDIR}/
scp /home/bastian/D1/registration/cubeslurm/${SLURM_JOB_ID}/.out ${OUTPUTDIR}/
echo "success"