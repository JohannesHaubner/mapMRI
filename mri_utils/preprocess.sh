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

MRI2FEMDATA=/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/mri2fem-dataset
TARGETDIR=${MRI2FEMDATA}/processed

mkdir -vp ${TARGETDIR}

subj1=abby
subj2=ernie
IMAGE=mri/brain.mgz
REGDIR=${TARGETDIR}/registered

mkdir -vp ${REGDIR}

IMG1=${MRI2FEMDATA}/freesurfer/${subj1}/${IMAGE}
IMG2=${MRI2FEMDATA}/freesurfer/${subj2}/${IMAGE}
REGIMAGE=${REGDIR}/${subj1}to${subj2}.mgz

if [ ! -f ${REGIMAGE} ]; then
    echo "File not found!"
    mri_robust_register --mov ${IMG1} --dst ${IMG2} --lta ${REGDIR}/${subj1}to${subj2}.lta \
    --mapmov ${REGIMAGE} --weights ${REGDIR}/${subj1}to${subj2}-weights.mgz --iscale --satit

    freeview ${IMG1} ${IMG2} ${REGDIR}/${subj1}to${subj2}.mgz ${REGDIR}/${subj1}to${subj2}-weights.mgz:colormap=heat
else
    echo "Found target file"
    echo ${REGIMAGE}
    echo "not running Freesurfer registration"
fi


