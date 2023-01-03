#!/bin/bash

ITERS=100

for ALPHA in 1e-2 1e-4 # 1e-6 1e-7 1e-3 1e-1 1e-5
do
    if [ -z ${jobid+x} ]; 
    then
        sentence=$(sbatch 2dsubmit.slurm ${ITERS} ${ALPHA})
    else
        # sentence=$(sbatch --dependency=afterany:${jobid} 2dsubmit.slurm ${ITERS} ${ALPHA})
        sentence=$(sbatch 2dsubmit.slurm ${ITERS} ${ALPHA})
    fi
    stringarray=($sentence)
    jobid=(${stringarray[3]});
    
    echo $jobid $sentence
done