"""
Written by Yngve Mardal Moe 

https://github.com/yngvem/brainsim


"""

from contextlib import contextmanager
from pathlib import Path
import numpy as np
# import SVMTK as svmtk


def transform_coords(coords, transform, inverse=False):
    if inverse:
        transform = np.linalg.inv(transform)
    coords = np.asarray(coords)
    if coords.ndim == 2:
        coords = coords.T
        coords = transform[:3, :3]@coords + transform[:3, 3, np.newaxis]
        coords = coords.T
    else:
        coords = transform[:3, :3] @ coords + transform[:3, 3]

    return coords


def get_cras_from_nifti(image_nii):
    """Get the center of the RAS coordinate system.

    Based on: https://neurostars.org/t/freesurfer-cras-offset/5587/2
    Which links to: https://github.com/nipy/nibabel/blob/d1518aa71a8a80f5e7049a3509dfb49cf6b78005/nibabel/freesurfer/mghformat.py#L630-L643
    """
    shape = np.array(image_nii.shape)
    center = shape / 2
    center_homogeneous = np.hstack((center, [1]))
    transform = image_nii.affine
    return (transform @ center_homogeneous)[:3]


def get_surface_ras_to_image_coordinates_transform(image_nii, surface_metadata=None):
    """Convert from freesurfer surface coordinates to scanner coordinates.

    Freesurfer uses (at least) three different coordinate systems, the RAS system,
    the surface RAS system (which is a shifted version of the RAS system) and the
    image coordinates. This function creates a transformation matrix that transforms
    the surface RAS system into image coordinates. To accomplish this, it uses the
    cras (center of RAS) property from the surface metadata to translate the surface
    RAS into the correct RAS coordinates. Then it uses the inverse of the image
    coordinate to RAS transformation to transform the RAS coordinates into image
    coordinates.

    The surface metadata is obtained from using the ``nibabel.freesurfer.read_geometry``
    function like this:

        points, faces, metadata = nib.freesurfer.read_geometry(path, read_metadata=True)
    
    The image coordinate transform is the affine transform in the nifti file whose image
    coordinates we want to transform into.

    If the metadata is not supplied, then it is assumed to be created from an image with
    the same coordinate systems as the given nifti file.

    Example
    -------


    >>> import nibabel as nib
    ... points, faces, metadata = nib.freesurfer.read_geometry("lh.pial", read_metadata=True)
    ... img = nib.load("T2W.nii")
    ... translation_matrix = get_surface_ras_to_image_coordinates_transform(img, metadata)
    ... translation_matrix = get_surface_ras_to_image_coordinates_transform(img)
    """
    translation_matrix = np.eye(4)
    if surface_metadata is not None:
        translation_matrix[:3, -1] = surface_metadata['cras']
    else:
        translation_matrix[:3, -1] = get_cras_from_nifti(image_nii)

    return np.linalg.inv(image_nii.affine)@translation_matrix


@contextmanager
def process_surface(input_file, output_file):
    surface = svmtk.Surface(str(input_file))
    yield surface
    surface.save(str(output_file))


def create_volume_mesh(input, output_file, resolution=16):
    if Path(input).is_dir():
        surface = [svmtk.Surface(str(p)) for p in input.iterdir()]
    else:
        surface = svmtk.Surface(str(input))

    domain = svmtk.Domain(surface)
    domain.create_mesh(resolution)

    domain.save(str(output_file))

