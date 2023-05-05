# Setup

```
git clone https://github.com/Baumi3004/Oscar-Image-Registration-via-Transport-Equation.git
cd Oscar-Image-Registration-via-Transport-Equation
conda env create -f dgregister-env.yml
conda activate dgregister-env
pip install -e .
```

# Data

Download the dataset from https://zenodo.org/communities/mri2fem and move the folder `freesurfer` to 
```
Oscar-Image-Registration-via-Transport-Equation/data/
```

# Image pre-processing

Requires FreeSurfer.
run 
```
$ bash scripts/1-image-preprocessing/preprocess.sh
```

This puts the pre-processed images to `Oscar-Image-Registration-via-Transport-Equation/data/normalized`

# Image registration



# Mesh generation


Requires FreeSurfer and SVMTK https://github.com/SVMTK/SVMTK.
Meshes used in the paper can be downloaded from https://github.com/bzapf/meshes.

Locate the meshes under `Oscar-Image-Registration-via-Transport-Equation/data/meshes/`


## Ventricular system mesh

Requires the manually edited freesurfer segmentation file for "Abby". 
Download from https://github.com/bzapf/meshes and move to `Oscar-Image-Registration-via-Transport-Equation/data/freesurfer/abby/reg-ventricles-w-aq.mgz`.


To create the ventricular system surface files, run
```
$ bash scripts/2-meshing/exctract-ventricles.sh
```
Then, meshing using

```
$ python scripts/2-meshing/make_ventricle_mesh.py
```



## Left hemisphere mesh

As described in Mardal et al. "Mathematical modeling of the human brain: from magnetic resonance images to finite element simulation" Springer 2022.
The resulting meshes can be downloaded from https://github.com/bzapf/meshes.


# Mesh registration


## Affine mesh registration
We perform the affine registration of the mesh manually. 
This is useful to visualize the affine-registered meshes together with the target image.

```
$ python scripts/2-meshing/register_brain_mesh.py
```

## Velocity-field mesh registration.

Assuming you run on a server with SLURM, run

```
$ mkdir -pv ./outputs/mrislurm/
$ sbatch 4-postprocessing/optimize3d.slurm 
```

Alternatively,
```
$ export IMG1=./data/normalized/cropped/cropped_abbytoernie_nyul.mgz
$ export IMG2=./data/normalized/cropped/cropped_ernie_brain_nyul.mgz
python3 -u ./scripts/3-optimization/Optimize3d.py --output_dir my_registration \
--input ${IMG1} --target ${IMG2}
```