#!/bin/bash


for NCOR in 1 4 10
do
sbatch coarsesubmit.slurm $NCOR
done
