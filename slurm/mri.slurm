#!/bin/bash
#SBATCH --job-name=mrireg
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=96
#SBATCH --exclusive
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=32
#################################################################
## SBATCH --partition=milanq
## milanq has 256 cpus per node
#
## SBATCH --partition=fpgaq
## fpgaq has 96 cpus per onde
#
#SBATCH --partition=defq
## SBATCH --partition=defq,milanq
## has 128 cpus per node
## SBATCH --partition=habanaq
#################################################################
#SBATCH --time 6-23:00:0
#SBATCH -o /global/D1/homes/bastian/registration/mrislurm/%j.out
#SBATCH --export=ALL
#SBATCH --mail-user=bastian@simula.no

eval "$(conda shell.bash hook)"
conda activate mri_inverse
LOGOUTFILE2=/home/bastian/D1/registration/mrislurm/${SLURM_JOB_ID}_log_python_srun.txt


set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

echo "starting to run"

#######################################################
## EXECUTE AFTER ANOTHER JOB IS DONE:
## sbatch --dependency=afterany: submit.slurm
#######################################################

GPATH=/home/bastian/Oscar-Image-Registration-via-Transport-Equation/
SPATH=${GPATH}scripts/optimization
LOGOUTFILE=/home/bastian/D1/registration/mrislurm/${SLURM_JOB_ID}_log_python_srun.txt

OUTPUTDIR=/home/bastian/D1/registration/smoothened-outputs/${SLURM_JOB_ID}/
IMG1=/home/bastian/D1/registration/mri2fem-dataset/normalized/cropped/cropped_abbytoernie_nyul.mgz
IMG2=/home/bastian/D1/registration/mri2fem-dataset/normalized/cropped/cropped_ernie_brain_nyul.mgz
LBFGS=200


NP=96
ITERS=100
ALPHA=1e-2

echo "NP=" ${NP}
echo "ITERS=" ${ITERS}
echo "ALPHA=" ${ALPHA}
echo "OUTPUTDIR=" $OUTPUTDIR
echo "IMG1=" ${IMG1}
echo "IMG2=" ${IMG2}

srun -n ${NP} --cpu-bind=rank python3 -u ${SPATH}/Optimize3d.py \
--logfile ${LOGOUTFILE2} --output_dir ${OUTPUTDIR} \
--input ${IMG1} --target ${IMG2} --max_timesteps ${ITERS} --alpha ${ALPHA} --lbfgs_max_iterations ${LBFGS} \
--smoothen_image --huber_delta 1 --timestepping RungeKutta --huber --memdebug --slurmid ${SLURM_JOB_ID} > ${LOGOUTFILE}

# --starting_state /home/bastian/D1/registration/normalized-outputs/447918/RK100A0.01LBFGS150/State_checkpoint.xdmf --readname CurrentState \
# --starting_state /home/bastian/D1/registration/normalized-forward/446152/RK100A0.01LBFGS0/Finalstate.xdmf --statename Finalstate \
# --starting_guess /home/bastian/D1/registration/normalized-outputs/446600/RK100A0.01LBFGS200/Control_checkpoint.xdmf --readname CurrentV \
# --forward \
# --starting_state /global/D1/homes/bastian/registration/normalized-outputs/446152/RKA0.01LBFGS100/State_checkpoint.xdmf --readname CurrentState \

# --starting_state /global/D1/homes/bastian/registration/normalized-outputs/446148/RKA0.01LBFGS10/State_checkpoint.xdmf --readname CurrentState \

# --starting_state /global/D1/homes/bastian/registration/normalized-outputs/445957/RKA0.01LBFGS10/State_checkpoint.xdmf --readname CurrentState \

# --starting_guess /home/bastian/D1/registration/normalized-outputs/445326/RKA0.01LBFGS50/Control_checkpoint.xdmf --readname CurrentV \
# --starting_state /global/D1/homes/bastian/registration/normalized-outputs/445806/RKA0.01LBFGS10/State_checkpoint.xdmf --readname CurrentState \

# cp -vr /global/D1/homes/bastian/registration/test/445806 /global/D1/homes/bastian/registration/normalized-outputs/


# --starting_guess /home/bastian/D1/registration/normalized-outputs/445326/RKA0.01LBFGS50/Control_checkpoint.xdmf --readname CurrentV \

# --starting_state /home/bastian/D1/registration/normalized-outputs/445457/RKA0.01LBFGS50/State_checkpoint.xdmf --readname CurrentState \

# --starting_guess /home/bastian/D1/registration/normalized-outputs/445326/RKA0.01LBFGS50/Control_checkpoint.xdmf --readname CurrentV \

# --starting_state /home/bastian/D1/registration/normalized-outputs/445326/RKA0.01LBFGS50/State_checkpoint.xdmf --readname CurrentState \
# --starting_state /home/bastian/D1/registration/normalized-outputs/445326/RKA0.01LBFGS50/State_checkpoint.xdmf --readname CurrentState \

# --starting_guess /home/bastian/D1/registration/normalized-outputs/445326/RKA0.01LBFGS50/Control_checkpoint.xdmf --readname CurrentV \
# --starting_state /home/bastian/D1/registration/normalized-outputs/445326/RKA0.01LBFGS50/State_checkpoint.xdmf --readname CurrentState \


echo "success"
