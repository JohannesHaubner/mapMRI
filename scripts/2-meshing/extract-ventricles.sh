#!/bin/bash
set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

# Un/comment the subject you want to process 


# # ABBY
# input="./data/freesurfer/abby/mri/reg-ventricles-w-aq.mgz"
# outputdir=./data/meshes/abby/affine_registered_ventricles/
# output=${outputdir}ventricles.stl
# output2="./data/meshes/abby/affine_registered_ventricles/ventricles_postproc.mgz"
# cp $input tmp.mgz


# ERNIE
input="./data/freesurfer/ernie/mri/wmparc.mgz"
outputdir=./data/meshes/ernie/ventricles/
output=${outputdir}ventricles.stl
output2="./data/meshes/ernie/ventricles/ventricles_postproc.mgz"
mri_binarize --i $input --ventricles --match 15 --o "tmp.mgz"




########################################################################

mkdir -pv $outputdir
num_smoothing=2
num_closing=2
V_min=100


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


rm tmp-ocn.lut
rm tmp.mgz
rm tmp-ocn.mgz
exit
