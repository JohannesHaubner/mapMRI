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

CODEDIR=/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation

MRI2FEMDATA=${CODEDIR}/mri2fem-dataset
TARGETDIR=${MRI2FEMDATA}/processed

mkdir -vp ${TARGETDIR}

subj1=abby
subj2=ernie
IMAGENAME=brain.mgz
IMAGE=mri/${IMAGENAME}

INPUTDIR=${TARGETDIR}/input
mkdir -vp ${INPUTDIR}

IMG1PATH=${INPUTDIR}/${subj1}
IMG2PATH=${INPUTDIR}/${subj2}

mkdir -pv ${IMG1PATH}
mkdir -pv ${IMG2PATH}

IMG1=${IMG1PATH}/${subj1}_${IMAGENAME}
IMG2=${IMG2PATH}/${subj2}_${IMAGENAME}

cp -v ${MRI2FEMDATA}/freesurfer/${subj1}/${IMAGE} ${IMG1}
cp -v ${MRI2FEMDATA}/freesurfer/${subj2}/${IMAGE} ${IMG2}

REGDIR=${TARGETDIR}/registered
mkdir -vp ${REGDIR}
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

IMG1=${REGIMAGE}
CROPDIR=${TARGETDIR}/cropped
mkdir -vp ${CROPDIR}

python ${CODEDIR}/mri_utils/mask_mri.py --images ${IMG1} ${IMG2} --targetfolder ${CROPDIR} --crop

IMG1=${REGIMAGE}
COARSECROPDIR=${TARGETDIR}/coarsecropped
mkdir -vp ${COARSECROPDIR}

python ${CODEDIR}/mri_utils/mask_mri.py --images ${IMG1} ${IMG2} --targetfolder ${COARSECROPDIR} --crop --coarsen