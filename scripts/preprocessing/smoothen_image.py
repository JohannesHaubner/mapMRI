import nibabel
import os
from scipy import ndimage

imagepath = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/normalized/cropped/"

outputpath = imagepath.replace("cropped", "smooth")

os.makedirs(outputpath, exist_ok=True)

files = [x for x in os.listdir(imagepath) if x.endswith(".mgz")]

for file in files:

    image = nibabel.load(imagepath + file)
    
    aff = image.affine
    
    image= image.get_fdata()

    smooth_image = ndimage.gaussian_filter(image, sigma=0.5)
    # smooth_image = ndimage.median_filter(image, size=2, origin=[-1, -1, -1])

    nibabel.save(nibabel.Nifti1Image(smooth_image, aff), outputpath + file.replace(".mgz", "_smooth.mgz"))

