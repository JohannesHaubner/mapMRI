#!/bin/bash
set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd 

echo "FreeSurfer configuration is required to run this script" 
if [ ! -z "${FREESURFER_HOME}" ];
then
   echo "FreeSurfer found"  
else 
   echo "FreeSurfer not found" 
   exit 
fi

export SUBJECTS_DIR=/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/freesurfer_copy

echo "SUBJECTS_DIR=" $SUBJECTS_DIR

mri_cvs_register --mov abby --template ernie --outdir ${SUBJECTS_DIR}/cvs