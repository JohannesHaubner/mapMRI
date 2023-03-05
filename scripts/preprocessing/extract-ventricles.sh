#!/bin/bash

# Input and output filenames

# 

# input="/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/freesurfer/abby/mri/wmparc.mgz"
# output="/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/meshes/ventricles/abby/ventricles.stl"
# output2="/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/meshes/ventricles/abby/ventricles.mgz"
wmparc=true

input="/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/normalized/registered/abbytoernie_ventricles.mgz"
outputdir=/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/meshes/registered_ventricles/abby/
output=${outputdir}ventricles.stl
output2="/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/meshes/registered_ventricles/abby/ventricles.mgz"
output3="/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/meshes/registered_ventricles/abby/lh.reg_ventricles"
output4="/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/meshes/registered_ventricles/abby/reg_ventricles"
wmparc=false

mkdir -pv $outputdir

# Also match the 4th ventricle and aqueduct?
include_fourth_and_aqueduct=true
if [ "$include_fourth_and_aqueduct" == true ]; then
    matchval="15"
else
    matchval="1"
fi
num_smoothing=2

# Other parameters
postprocess=true
num_closing=2
V_min=100

if [ "$postprocess" == true ]; then
    if [ "$wmparc" == true ]; then
    mri_binarize --i $input --ventricles \
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
    mri_binarize --i "tmp.mgz" \
	         --match 1 \
	         --surf-smooth $num_smoothing \
	         --surf $output3

    mri_binarize --i "tmp.mgz" \
	         --match 1 \
	         --surf-smooth $num_smoothing \
	         --surf $output4

    rm tmp-ocn.lut
    rm tmp.mgz
    rm tmp-ocn.mgz
    exit
fi

mri_binarize --i $input --ventricles \
	--match $matchval \
	--surf-smooth $num_smoothing \
	--surf $output
