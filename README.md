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

# Meshing

## Ventricular system mesh

## Left hemisphere mesh

As described in Mardal et al. "Mathematical modeling of the human brain: from magnetic resonance images to finite element simulation" Springer 2022.

## Mesh registration

We perform the affine registration of the mesh manually. This is useful to visualize the affine-registered meshes.

```
$ python scripts/2-meshing/register_brain_mesh.py
```