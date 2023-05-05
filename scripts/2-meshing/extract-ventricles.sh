#!/bin/bash

# Input and output filenames

# ABBY
# input="./Oscar-Image-Registration-via-Transport-Equation/data/meshes/reg-aqueduct/abby/reg-ventricles-w-aq.mgz"
# outputdir=./Oscar-Image-Registration-via-Transport-Equation/data/meshes/reg-aqueduct/abby/
# output=${outputdir}ventricles.stl
# output2="./Oscar-Image-Registration-via-Transport-Equation/data/meshes/reg-aqueduct/abby/ventricles_postproc.mgz"

# ERNIE
input="/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/data/freesurfer/ernie/mri/wmparc.mgz"
outputdir=/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/data/meshes/aqueduct/ernie/
output=${outputdir}ventricles.stl
output2="/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/data/meshes/aqueduct/ernie/ventricles_postproc.mgz"


# Also match the 4th ventricle and aqueduct?
include_fourth_and_aqueduct=true

# Other parameters
postprocess=true
wmparc=true


mkdir -pv $outputdir


if [ "$include_fourth_and_aqueduct" == true ]; then
    matchval="15"
else
    matchval="1"
fi
num_smoothing=2


num_closing=2
V_min=100

if [ "$postprocess" == true ]; then
    if [ "$wmparc" == true ]; then
    mri_binarize --i $input --ventricles \
    	    --match $matchval \
	         --o "tmp.mgz"
    
    else
    cp $input tmp.mgz
    fi
    mri_volcluster --in "tmp.mgz" \
	           --thmin 1 \
	           --minsize $V_min \
	           --ocn "tmp-ocn.mgz"
    
    mri_binarize --i "tmp-ocn.mgz" \
	         --match 1 \
	         --o "tmp.mgz"
    
    mri_morphology "tmp.mgz" \
	           close $num_closing "tmp.mgz"
    
    mri_binarize --i "tmp.mgz" \
	         --match 1 \
	         --surf-smooth $num_smoothing \
	         --surf $output --o $output2

    # mri_binarize --i "tmp.mgz" \
	#          --match 1 \
	#          --surf-smooth $num_smoothing \
	#          --surf $output3

    # mri_binarize --i "tmp.mgz" \
	#          --match 1 \
	#          --surf-smooth $num_smoothing \
	#          --surf $output4

    rm tmp-ocn.lut
    rm tmp.mgz
    rm tmp-ocn.mgz
    exit
fi

mri_binarize --i $input --ventricles \
	--match $matchval \
	--surf-smooth $num_smoothing \
	--surf $output
