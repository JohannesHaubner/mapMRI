import numpy as np
import nibabel
from scipy import ndimage


inputpath = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/mridata_3d/"

generic_affine = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
    ])

imagenames = ["091registeredto205", "205_cropped"]

npad = 4

def pad_with(vector, pad_width, iaxis, kwargs):

    pad_value = kwargs.get('padder', 0)

    vector[:pad_width[0]] = pad_value

    vector[-pad_width[1]:] = pad_value



for imgname in imagenames:

    img = nibabel.load(inputpath + imgname + ".mgz").get_fdata()

    # padded_img = np.pad(img, ((npad, npad), (npad, npad), (npad, npad)), mode='constant', constant_values=(0, 0, 0))

    padded_img = np.pad(img, npad, pad_with)

    result = ndimage.zoom(padded_img, 0.5)

    print("input shape", img.shape, "padded img shape", padded_img.shape, "new img shape", result.shape)

    nibabel.save(nibabel.Nifti1Image(padded_img, affine=generic_affine), inputpath + imgname + "_padded.mgz")
    nibabel.save(nibabel.Nifti1Image(result, affine=generic_affine), inputpath + imgname + "_padded_coarsened.mgz")
