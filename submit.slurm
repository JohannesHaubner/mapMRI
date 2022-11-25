#!/bin/bash
#SBATCH --job-name=register
#SBATCH --cpus-per-task=1 --nodes=4
## SBATCH --partition=milanq
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


OUTFOLDERNAME=testKrylov
srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} --output_dir ${OUTPUTDIR} \
--lbfgs_max_iterations 1 --solver krylov > ${LOGOUTFILE}

# OUTFOLDERNAME=test
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} \
# --lbfgs_max_iterations 1 --smoothen --solver krylov > ${LOGOUTFILE}

# OUTFOLDERNAME=testLu
# srun -n 4 python3 -u ${SPATH}/Optimize3d.py --outfolder ${OUTFOLDERNAME} \
# --lbfgs_max_iterations 1 --smoothen --solver lu > ${LOGOUTFILE}
 
OUTPATH=${OUTPUTDIR}${OUTFOLDERNAME}/
cp -v ${LOGOUTFILE} ${OUTPATH}

# Assembled A in Preconditioning()
# type of A <class 'dolfin.cpp.la.Matrix'> <dolfin.cpp.la.Matrix object at 0x155541f7f4a0>
# type of self.solver <class 'fenics_adjoint.petsc_krylov_solver.PETScKrylovSolver'>
# Created Krylov solver in Preconditioning()


# Assembled A in PreconditioningBlock()
# type of A <class 'dolfin.cpp.la.Vector'> <dolfin.cpp.la.Vector object at 0x15552f81dea0>
# type of self.solver <class 'fenics_adjoint.petsc_krylov_solver.PETScKrylovSolver'>
# Created Krylov solver in PreconditioningBlock()


echo "success"