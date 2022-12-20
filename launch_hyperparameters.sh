#!/bin/bash

ALPHA=1e-0
for MAXIITER in 64 128 256 512
do
    if [ -z ${jobid+x} ]; 
    then
        sentence=$(sbatch submit.slurm ${MAXIITER} ${ALPHA})
    else
        sentence=$(sbatch --dependency=afterany:${jobid} submit.slurm ${MAXIITER} ${ALPHA})
    fi
    stringarray=($sentence)
    jobid=(${stringarray[3]});
    
    echo $jobid $sentence
done
