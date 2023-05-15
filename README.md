# mapMRI 

Code repository for the manuscript

> Bastian Zapf, Johannes Haubner, Lukas Baumgärtner, Stephan Schmidt: Medical Image Registration using optimal control of a linear hyperbolic transport equation with a DG discretization, arXiv:2305.03020 (2023)

## Setup

```
git clone https://github.com/JohannesHaubner/mapMRI.git
cd mapMRI
conda env create -f dgregister-env.yml
conda activate dgregister-env
pip install -e .
```

#### Note: All scripts are assumed to be run from the top level of this repository.

## Data

We use the publicly available dataset from https://zenodo.org/communities/mri2fem.
From this dataset, we manually edit a segmentation file and create meshes as outlined in the manuscript.
For convenience, the masked MRI, edited segmentation files, and meshes reported upon in the manuscript can be found in this repository under
```
mapMRI/data/
```
These files are sufficient to reproduce the reported results. 


## Image pre-processing

If you want to start from scratch and perform the preprocessing steps, download the dataset from https://zenodo.org/communities/mri2fem and move the folder `FreeSurfer` to `mapMRI/data/`.
Requires FreeSurfer and https://github.com/jcreinhold/intensity-normalization (`pip install intensity-normalization`).

run 
```
bash scripts/1-image-preprocessing/preprocess.sh
```

This puts the pre-processed images to `mapMRI/data/normalized`

# Image registration

To reproduce the image registration results reported in the manuscript, run the following commands
```
export IMG1=./data/normalized/cropped/cropped_abbytoernie_nyul.mgz
export IMG2=./data/normalized/cropped/cropped_ernie_brain_nyul.mgz
python3 ./scripts/3-optimization/Optimize3d.py --output_dir ./outputs/my_registration_1 --input ${IMG1} --target ${IMG2} --lbfgs_max_iterations 100 
```

Improve upon the first registration by a second velocity based transform:

```
python3 ./scripts/3-optimization/Optimize3d.py --output_dir ./outputs/my_registration_2 --input ${IMG1} --target ${IMG2} --lbfgs_max_iterations 151
--starting_state my_registration_1/State_checkpoint.xdmf 
```

Improve upon the second registration by a third velocity based transform:

```
python3 ./scripts/3-optimization/Optimize3d.py --output_dir ./outputs/my_registration_3 --input ${IMG1} --target ${IMG2} --lbfgs_max_iterations 151 
--starting_state my_registration_2/State_checkpoint.xdmf
```


Improve upon the registration by a less smooth velocity based transform by tweaking the velocity transform hyperparameters:

```
python3 ./scripts/3-optimization/Optimize3d.py --output_dir ./outputs/my_registration_4 --input ${IMG1} --target ${IMG2} --lbfgs_max_iterations 114
--starting_state my_registration_3/State_checkpoint.xdmf \
--alpha 0.5 --beta 0.5
```

On a server with SLURM, you can use scripts/3-optimization/optimize3d.slurm.


## Mesh generation

The meshes used in the paper are located under `mapMRI/data/meshes/`.
Alternatively, they can be generated from MRI as described below.

### Ventricular system mesh

Requires the manually edited FreeSurfer segmentation file for "Abby" found under `mapMRI/data/freesurfer/abby/reg-ventricles-w-aq.mgz`.


To create the ventricular system surface files, run
```
bash scripts/2-meshing/exctract-ventricles.sh
```
Then, meshing using

```
python scripts/2-meshing/make_ventricle_mesh.py
```

### Left hemisphere mesh

As described in Mardal et al. "Mathematical modeling of the human brain: from magnetic resonance images to finite element simulation" Springer 2022.
The scripts to generate these meshes are found at https://zenodo.org/communities/mri2fem.


## Mesh registration


### Affine mesh registration
We perform the affine registration of the mesh manually. 
This is useful to visualize the affine-registered meshes together with the target image.

```
python scripts/2-meshing/register_brain_mesh.py
```

The aqueduct mesh for "Abby" was created from a registered ventricles file, so this mesh does not need to be manually pre-registered.

### Velocity-field mesh registration.

First, create FEniCS files that describe how mesh coordinates should be mapped:
```
python3 ./scripts/4-postprocessing/transform_coordinates.py --folder ./outputs/my_registration_1/
python3 ./scripts/4-postprocessing/transform_coordinates.py --folder ./outputs/my_registration_2/
python3 ./scripts/4-postprocessing/transform_coordinates.py --folder ./outputs/my_registration_3/
python3 ./scripts/4-postprocessing/transform_coordinates.py --folder ./outputs/my_registration_4/

```

Then, deform the mesh:
```
export MESHFILE=./data/meshes/abby/manually_registered_brain_mesh/output/abby_registered_brain_mesh.xml
python3 ./scripts/4-postprocessing/transform_mesh.py \
--folders ./outputs/my_registration_1 ./outputs/my_registration_2 ./outputs/my_registration_3 ./outputs/my_registration_4 \
--input_meshfile ${MESHFILE} \
--meshoutputfolder  ./outputs/meshes/deformed_brain_mesh/
```

See also the SLURM scripts in ./scripts/4-postprocessing/.


# Visualization

MR images can be viewed in freeview. 
For visualization of images and meshes, we used paraview. 

To create files that can be viewed in freeview from registration output, run 
```
python3 scripts/5-utils/currentState_to_mgz.py \
./outputs/my_registration_1
```
This creates the file CurrentState.mgz which can be viewed in freeview.

After this, to convert the output such that it can be viewed in paraview (large file!) run
```
python3 scripts/5-utils/image2pv.py \
./outputs/my_registration_1/CurrentState.mgz
```

To create the plot showing the reduction of L2-error, see
```
python3 scripts/4-postprocessing-paperlossplot.py
```


See also the script `scripts/5-utils/compare-DGreg-to-cvs.py` which loads the DG-registered and freesurfer-cvs registered files to freeview.
