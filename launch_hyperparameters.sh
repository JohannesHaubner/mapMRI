#!/bin/bash

ITERS=1000

for ALPHA in 1e-6 1e-8
#1e-0 1e-2 1e-4 1e-6
do
    if [ -z ${jobid+x} ]; 
    then
        sentence=$(sbatch 2dsubmit.slurm ${ITERS} ${ALPHA})
    else
        sentence=$(sbatch --dependency=afterany:${jobid} 2dsubmit.slurm ${ITERS} ${ALPHA})
        # sentence=$(sbatch 2dsubmit.slurm ${ITERS} ${ALPHA})
    fi
    stringarray=($sentence)
    jobid=(${stringarray[3]});
    
    echo $jobid $sentence
done