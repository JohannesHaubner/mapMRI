# Information about how to compare the registration to freesurfer cvs-register
We worked with brain.mgz, freesurfer cvs-register with norm.mgz.
To allow for quantitative comparison between the two approaches, we apply the obtained velocity transforms to norm.mgz. 
To apply our transform to norm.mgz (the image used by FreeSurfer cvs-register), run the following commands:

First, affine registration:

mri_vol2vol --mov ../data/freesurfer/abby/mri/norm.mgz 
--targ ../data/freesurfer/ernie/mri/norm.mgz --o normalized/registered/abby2ernie_norm.mgz 
--lta normalized/registered/abbytoernie.lta

Then crop the images:
```
conda activate dgregister-env
CODEDIR=.
MRI2FEMDATA=${CODEDIR}/data
TARGETDIR=${MRI2FEMDATA}/normalized
CROPDIR=${TARGETDIR}/cropped_norm
mkdir -pv ${CROPDIR}
python ${CODEDIR}/scripts/1-image-preprocessing/mask_mri.py --images ./data/normalized/registered/abby2ernie_norm.mgz --targetfolder ${CROPDIR} \
--crop --box ./data/normalized/cropped/box.npy
```


Then run the transforms as in `scripts/4-postprocessing/transport-norm-image.slurm`

Next, store the deformed images in mgz format so they can be loaded to freeview:
```
python scripts/5-utils/currentState_to_mgz.py ./outputs/my_registration_3/norm_registered/ --xdmffile Finalstate.xdmf --readname Finalstate
python scripts/5-utils/currentState_to_mgz.py ./outputs/my_registration_4/norm_registered/ --xdmffile Finalstate.xdmf --readname Finalstate
```


The comparison figure is created in an axial slice at voxels
125, 110, 111
To view the images in freeview, run 
```
python scripts/5-utils/compare-DGreg-to-cvs.py
```