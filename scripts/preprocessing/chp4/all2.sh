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

echo "Checking if path to mri2fem dataset is set" 
if  [ ! -z "${MRI2FEMDATA}" ]; 
then
   echo "mri2fem dataset found"
else
   echo "mri2fem dataset not found"
   echo "Run setup in mri2fem-dataset folder" 
   echo "source Setup_mri2fem_dataset.sh" 
   exit
fi

IDNAME=ernie

OUTDIR=./outs/$IDNAME
mkdir -pv $OUTDIR

SCRIPTDIR=$PWD
cd $OUTDIR

# copy one surface file for later use
cp ${MRI2FEMDATA}/freesurfer/${IDNAME}/surf/lh.pial .

# Convert left hemisphere (lh) surfaces to STL


mris_convert ${MRI2FEMDATA}/freesurfer/${IDNAME}/surf/lh.pial ./lh.pial.stl
mris_convert ${MRI2FEMDATA}/freesurfer/${IDNAME}/surf/lh.white ./lh.white.stl


# Convert left hemisphere (rh) surfaces to STL
mris_convert ${MRI2FEMDATA}/freesurfer/${IDNAME}/surf/rh.pial ./rh.pial.stl
mris_convert ${MRI2FEMDATA}/freesurfer/${IDNAME}/surf/rh.white ./rh.white.stl

# # # Use scripts from chp3 to remesh and smoothen.
# # # Rename stl files lh.pial.stl and lh.white.stl again

# # # # Generate gray-white mesh
# # # python3 two-domain-tagged.py

# # # # Convert to paraview friendly format
# # # meshio-convert ${IDNAME}-gw.mesh ${IDNAME}-gw.vtu
 
# # # ./extract-ventricles.sh

## Set postprocess=true and run again:
cp ${MRI2FEMDATA}/freesurfer/${IDNAME}/mri/wmparc.mgz .
${SCRIPTDIR}/extract-ventricles.sh


# # # cp freesurfer/${IDNAME}/surf/rh.pial .
# # # cp freesurfer/${IDNAME}/surf/rh.white .
  


# Use scripts from chp3 to remesh and smoothen.
# Rename stl files rh.pial.stl and rh.white.stl again
RES=32
python3 ${SCRIPTDIR}/fullbrain-five-domain.py --resolution ${RES} --name ${IDNAME}

outname=${IDNAME}${RES}

meshio-convert ${outname}.mesh ${outname}.xml
meshio-convert ${outname}.xml ${outname}.xdmf
# paraview ${outname}.xdmf


RES=64
python3 ${SCRIPTDIR}/fullbrain-five-domain.py --resolution ${RES} --name ${IDNAME}

outname=${IDNAME}${RES}

meshio-convert ${outname}.mesh ${outname}.xml
meshio-convert ${outname}.xml ${outname}.xdmf