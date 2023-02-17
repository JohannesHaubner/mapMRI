#!/bin/bash
#SBATCH --job-name=cube
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
## SBATCH --exclusive
## SBATCH --nodes=1
## SBATCH --ntasks-per-node=32
#################################################################
## SBATCH --partition=milanq
## milanq has 256 cpus per node
#
#SBATCH --partition=fpgaq
## fpgaq has 96 cpus per onde
#
## SBATCH --partition=defq
## has 128 cpus per node
## SBATCH --partition=habanaq
#################################################################
#SBATCH --time 3-00:00:0
#SBATCH -o /global/D1/homes/bastian/registration/cubeslurm/%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=bastian@simula.no

set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

echo "starting to run"

module purge
module load singularity-ce
module load mpich-3.3.2

export SINGULARITYENV_CXX=/usr/bin/c++
export SINGULARITY_DOCKER_USERNAME=johanneshaubner
export SINGULARITY_DOCKER_PASSWORD=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

export SINGULARITY_BIND="/global/D1/homes/bastian:/home/bastian/d1"


#######################################################
## EXECUTE AFTER ANOTHER JOB IS DONE:
## sbatch --dependency=afterany: submit.slurm
#######################################################
GPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/
SPATH=${GPATH}scripts/optimization
LOGOUTFILE2=/home/bastian/d1/registration/cubeslurm/${SLURM_JOB_ID}_log_python_srun.txt
LOGOUTFILE=/home/bastian/D1/registration/cubeslurm/${SLURM_JOB_ID}_log_python_srun.txt


OUTPUTDIR=/home/bastian/d1/registration/cuberegistration/${SLURM_JOB_ID}/
DPATH=/home/bastian/d1/registration
DATAPATH=${DPATH}/testdata_3d/
IMG1=${DATAPATH}input.mgz
IMG2=${DATAPATH}newtarget.mgz


NP=4

echo "NP=" ${NP}

ITERS=50
ALPHA=1e0
LBFGS=50

echo "ITERS=" ${ITERS}
echo "ALPHA=" ${ALPHA}
echo "OUTPUTDIR=" $OUTPUTDIR
echo "IMG1=" ${IMG1}
echo "IMG2=" ${IMG2}



srun -n ${NP} singularity exec docker://ghcr.io/johanneshaubner/shapeopt:latest \
python3 -u ${SPATH}/Optimize3d.py \
--logfile ${LOGOUTFILE2} --output_dir ${OUTPUTDIR} \
--input ${IMG1} --target ${IMG2} --max_timesteps ${ITERS} --alpha ${ALPHA} --lbfgs_max_iterations ${LBFGS} \
--timestepping RungeKutta --huber --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}

echo "success"
