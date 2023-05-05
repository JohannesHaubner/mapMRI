# Setup

```
git clone https://github.com/Baumi3004/Oscar-Image-Registration-via-Transport-Equation.git
cd Oscar-Image-Registration-via-Transport-Equation
conda env create -f dgregister-env.yml
conda activate dgregister-env
pip install -e .
```

# Data

Download the dataset from https://zenodo.org/communities/mri2fem and extract into
```
Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset
```

# Image pre-processing

Requires FreeSurfer.
run 
```
$ bash scripts/1-image-preprocessing/preprocess.sh
```

# Image registration



# Mesh generation


Requires FreeSurfer and SVMTK https://github.com/SVMTK/SVMTK.
Meshes used in the paper can be downloaded from https://github.com/bzapf/meshes.
Should be put under Oscar-Image-Registration-via-Transport-Equation/data/meshes/


## Ventricular system mesh

Requires the manually edited freesurfer segmenation file. Download from https://github.com/bzapf/meshes
and put

Oscar-Image-Registration-via-Transport-Equation/data/freesurfer/abby/reg-ventricles-w-aq.mgz

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

We perform the affine registration of the mesh manually. This is useful to visualize the affine-registered meshes.

```
$ python scripts/2-meshing/register_brain_mesh.py
```


```

```