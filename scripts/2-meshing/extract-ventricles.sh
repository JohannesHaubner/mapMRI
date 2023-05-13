#!/bin/bash
set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

# Input and output filenames

# ABBY
input="./data/freesurfer/abby/mri/reg-ventricles-w-aq.mgz"
outputdir=./data/meshes/reg-aqueduct/abby/
output=${outputdir}ventricles.stl
output2="./data/meshes/reg-aqueduct/abby/ventricles_postproc.mgz"

# # ERNIE
# input="./data/freesurfer/ernie/mri/wmparc.mgz"
# outputdir=./data/meshes/aqueduct/ernie/
# output=${outputdir}ventricles.stl
# output2="./data/meshes/aqueduct/ernie/ventricles_postproc.mgz"


mkdir -pv $outputdir


num_smoothing=2


num_closing=2
V_min=100

cp $input tmp.mgz

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
