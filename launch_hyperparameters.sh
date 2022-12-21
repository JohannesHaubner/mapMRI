#!/bin/bash



# # parser.add_argument("--newTransport",type=str, choices=["true", "false"])
# # parser.add_argument("--alpha", default=1e-4, type=float)
# # parser.add_argument("--regularize", choices=["L2control", "velocity"])
# # parser.add_argument("--loading", choices=["Pic2FEN", "MRI2FEM"])
# # parser.add_argument("--normalize", type=str, choices=["true", "false"])
# python -u /home/bastian/Oscar-Image-Registration-via-Transport-Equation/Optimize.py --maxiter 10 --maxlbfs 1 \
# --newTransport $1 --alpha $2 --regularize $3 --loading $4 --normalize $5 > ${LOGOUTFILE}

ALPHA=1e-6
newTransport=false


for regularize in 'L2control' 'velocity'
do
for loading in "Pic2FEN" "MRI2FEM"
do
for normalize in true false
do
    if [ -z ${jobid+x} ]; 
    then
        sentence=$(sbatch debug.slurm ${newTransport} ${ALPHA} ${regularize} ${loading} ${normalize})
    else
        sentence=$(sbatch --dependency=afterany:${jobid} debug.slurm ${newTransport} ${ALPHA} ${regularize} ${loading} ${normalize})
    fi
    stringarray=($sentence)
    jobid=(${stringarray[3]});
    
    echo $jobid $sentence
done
done
done