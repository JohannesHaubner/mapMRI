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

MRI2FEMDATA=${CODEDIR}/registration/mri2fem-dataset
TARGETDIR=${MRI2FEMDATA}/normalized
normdir=${TARGETDIR}/nyul_normalized

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
    --mapmov ${REGIMAGE} --weights ${REGDIR}/${subj1}to${subj2}-weights.mgz --iscale --satit --vox2vox --maxit 20

    freeview ${IMG1} ${IMG2} ${REGDIR}/${subj1}to${subj2}.mgz ${REGDIR}/${subj1}to${subj2}-weights.mgz:colormap=heat
else
    echo "Found target file"
    echo ${REGIMAGE}
    echo "not running Freesurfer registration"
fi


# ### Normalization


img1=${REGIMAGE%.*}
img2=${IMG2%.*}

mri_convert $img1.mgz $img1.nii --in_orientation RAS --out_orientation RAS
mri_convert $img2.mgz $img2.nii  --in_orientation RAS --out_orientation RAS


tmpdir=${TARGETDIR}/tmp
mkdir -pv ${tmpdir}

cp -v ${img1}.nii ${tmpdir}/
cp -v ${img2}.nii ${tmpdir}/

maskdir=${TARGETDIR}/masks
mkdir -pv ${maskdir}

img1=$(basename $img1)
img2=$(basename $img2)

# make masks using mri_binarize
mri_binarize --i ${tmpdir}/${img1}.nii --min 1 --o ${maskdir}/${img1}.nii
mri_binarize --i ${tmpdir}/${img2}.nii --min 1 --o ${maskdir}/${img2}.nii

mri_morphology ${maskdir}/${img1}.nii fill_holes 4 ${maskdir}/${img1}.nii
mri_morphology ${maskdir}/${img1}.nii dilate 1 ${maskdir}/${img1}.nii

mri_morphology ${maskdir}/${img2}.nii fill_holes 4 ${maskdir}/${img2}.nii
mri_morphology ${maskdir}/${img2}.nii dilate 1 ${maskdir}/${img2}.nii

mkdir -pv $normdir

rm -f ${normdir}/*

nyul-normalize ${tmpdir}/ -o ${normdir} -vv -m ${maskdir}


cp -r ${normdir} ${TARGETDIR}/nyul_normalized_oriented

rm -r ${tmpdir}
#############


# make masks using mri_binarize
files=$(find ${normdir} -type f)

maskdir2=${TARGETDIR}/masks2
mkdir -pv ${maskdir2}

for file in ${files}; do
    fname=${file%.*}
    filename_without_path=$(basename $file)
    filename_without_path=${filename_without_path%.*}
    echo $fname
    echo $filename_without_path
    

    mri_binarize --i ${fname}.nii --min 1 --o ${maskdir2}/${filename_without_path}.nii
    mri_morphology ${maskdir2}/${filename_without_path}.nii fill_holes 4 ${maskdir2}/${filename_without_path}.nii
    mri_morphology ${maskdir2}/${filename_without_path}.nii dilate 1 ${maskdir2}/${filename_without_path}.nii


    mri_convert ${fname}.nii ${fname}.mgz --in_orientation RAS --out_orientation RAS
    mri_mask ${fname}.mgz ${maskdir2}/${filename_without_path}.nii ${fname}.mgz
    rm -v ${fname}.nii

done


files=$(find ${normdir} -type f)


CROPDIR=${TARGETDIR}/cropped
mkdir -vp ${CROPDIR}

python ${CODEDIR}/scripts/preprocessing/mask_mri.py --images ${files} --targetfolder ${CROPDIR} --crop

COARSECROPDIR=${TARGETDIR}/coarsecropped
mkdir -vp ${COARSECROPDIR}

python ${CODEDIR}/scripts/preprocessing/mask_mri.py --images ${files} --targetfolder ${COARSECROPDIR} --crop --coarsen

python ${CODEDIR}/scripts/imageinfo.py --imagedir ${CROPDIR}