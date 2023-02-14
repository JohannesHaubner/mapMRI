#!/bin/bash
set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

cd /home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/processed/normalized/

img1=/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/processed/normalized/affine_registered/cropped_abbytoernie_affine
img2=/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/processed/normalized/affine_registered/cropped_ernie_brain

mkdir -pv inputs

mri_convert $img1.mgz $img1.nii --in_orientation RAS --out_orientation RAS
mri_convert $img2.mgz $img2.nii  --in_orientation RAS --out_orientation RAS

cp -v $img1.nii ./inputs/
cp -v $img2.nii ./inputs/

mkdir -pv outputs



nyul-normalize ./inputs -o ./outputs -vv