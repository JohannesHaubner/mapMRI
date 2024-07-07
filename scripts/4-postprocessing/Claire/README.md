## Register and Compare with Claire 

Requires Claire and Clairetools (https://github.com/andreasmang/claire) at `$CLAIRE` and `$CLAIRETOOLS` as well as Freesurfer at `$FREESURFER_HOME`

To register with Claire, use Freesurfer to convert to `nii.gz` format. This has to be run after the preprocessing skript:

```
$FREESURFER_HOME/bin/mri_convert ./../../../data/normalized/nyul-normalized/cropped_abbytoernie_nyul.mgz   ./cropped_abbytoernie_nyul.nii.gz
$FREESURFER_HOME/bin/mri_convert ./../../../data/normalized/nyul-normalized/cropped_ernie_brain_nyul.mgz   ./cropped_ernie_brain_nyul.nii.gz
```
Now the registration with Claire can be performed. We used
```
mpirun -np 20 $CLAIRE -mr ./cropped_ernie_brain_nyul.nii.gz -mt ./cropped_abbytoernie_nyul.nii.gz -x ./output_ -velocity -maxit 1000 -beta 1e-3 -gabs 1e-8 -opttol 1e-6 -krylovmaxit 5000 -betacont 7.75e-08
```
This creates three `output_velocity_field-x*.nii.gz` files. Now convert the cropped/normalized image to transform according to these velocity fields.
```
$FREESURFER_HOME/bin/mri_convert ./../../../data/normalized/cropped_norm/cropped_abby2ernie_norm.mgz ./cropped_abby2ernie_norm.nii.gz
```
Now use `$CLAIRETOOLS` to transfrom these by
```
$CLAIRETOOLS -v1 output_velocity-field-x1.nii.gz -v2 output_velocity-field-x2.nii.gz -v3 output_velocity-field-x3.nii.gz -ifile ./cropped_abby2ernie_norm.nii.gz -xfile ./cropped_abby2ernie_norm_transported.nii.gz -deformimage
```
and convert them back using again Freesurfer
```
$FREESURFER_HOME/bin/mri_convert ./cropped_abby2ernie_norm_transported.nii.gz ./cropped_abby2ernie_norm_transported.mgz
```
We copy the cropping box and the background image to this directory
```
cp ./../../../data/freesurfer/ernie/mri/norm.mgz ./norm_ernie.mgz
cp ./../../../data/normalized/cropped/box.npy ./box.npy
```
Afterwards we can compute the turkey-errors and view the result in Freesurfer with the scrips
```
python3 compare_claire_in_freesurfer.py
```



