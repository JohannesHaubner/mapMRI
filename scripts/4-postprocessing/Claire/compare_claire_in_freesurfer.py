import os
import numpy as np
import nibabel
from dgregister.helpers import cut_to_box, get_bounding_box_limits
from dgregister.helpers import crop_to_original



box = np.load("./box.npy")
space = 0
pad = 2

def tukey(x,c=255):

    return np.mean(np.where(np.abs(x) <= c, c ** 2 / 2 * (1 - (1 - x ** 2 / c ** 2) ** 3), c ** 2 / 2))

inputimage_path = "./norm_ernie.mgz"
background_path = "./norm_ernie.mgz"

#currentstate_path = "./abbie_transported8.mgz"
currentstate_path = "./cropped_abby2ernie_norm_transported.mgz"


#target image 
inputimage = nibabel.load(inputimage_path)
inaff = inputimage.affine
inputimage = inputimage.get_fdata()

#filled_image_input = crop_to_original(orig_image=np.zeros((256, 256, 256)), cropped_image=inputimage, box=box, space=space, pad=pad)
filled_image_input = inputimage

inputnii = nibabel.Nifti1Image(filled_image_input, inaff)
inputimage_oriented = inputimage_path.replace(".mgz", "_oriented.mgz")
nibabel.save(inputnii, inputimage_oriented)


# registered image
stateimage = nibabel.load(currentstate_path)
stateaff = stateimage.affine
stateimage = stateimage.get_fdata()

filled_image_state = crop_to_original(orig_image=np.zeros((256, 256, 256)), cropped_image=stateimage, box=box, space=space, pad=pad)

statenii = nibabel.Nifti1Image(filled_image_state, inaff)
stateimage_oriented = currentstate_path.replace(".mgz", "_oriented.mgz")
nibabel.save(statenii, stateimage_oriented)


mydiff = np.abs(filled_image_input-filled_image_state)
#mydiff = np.abs(filled_image_state)


mydiffnii = nibabel.Nifti1Image(mydiff, inaff)
diffpath =  currentstate_path.replace(".mgz", "_diff.mgz")
nibabel.save(mydiffnii, diffpath)

background = nibabel.load(background_path)
backaff = background.affine
background = background.get_fdata()
filled_image_input = background
#filled_image_input = filled_image_input/np.max(filled_image_input)*255

backgroundnii = nibabel.Nifti1Image(filled_image_input, inaff)
background_oriented = background_path.replace(".mgz", "_oriented.mgz")
nibabel.save(backgroundnii, background_oriented)




import matplotlib.pyplot as plt

print("Tukey difference between target and 3-velocity transform")
print(tukey(inputnii.get_fdata()-statenii.get_fdata(), c=255))
print("Tukey difference between target and 3-velocity transform")
print(tukey(inputnii.get_fdata()-statenii.get_fdata(), c=110))
print("Tukey difference between target and 3-velocity transform")
print(tukey(inputnii.get_fdata()-statenii.get_fdata(), c=50))

command = "$FREESURFER_HOME/bin/freeview --slice 125 110 111 --viewport axial"
# command += " "
#command += normalized_input_oriented
# command += " "
# command += currentstate
command += " --colormap Heat:heatscale=10,30,50 -v " + diffpath
#command += " "
#command += " --colormap Heat:heatscale=10,30,50 -v " + diffpath2 
#command += " "
command += " --colormap Grayscale "
command += inputimage_oriented
command += " "

command += " --colormap Grayscale "
command += background_oriented
command += " "
# command += freesruferimage
#command += " --colormap Heat:heatscale=10,30,50 -v " + fsdiffpath
os.system(command)
