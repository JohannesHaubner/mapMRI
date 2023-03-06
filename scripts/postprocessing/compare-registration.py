import os, pathlib
import numpy as np
import nibabel
from dgregister.helpers import view
p = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/"

command = "freeview" 
inputimage = p + "mri2fem-dataset/normalized/input/ernie/ernie_brain.mgz"

normalized_input = p + "mri2fem-dataset/normalized/nyul_normalized/ernie_brain_nyul.mgz"
currentstate = p + "normalized-outputs/447918/RK100A0.01LBFGS150/CurrentState.mgz"

inaff = nibabel.load(inputimage).affine

norm_input = nibabel.load(normalized_input).get_fdata()

normnii = nibabel.Nifti1Image(norm_input, inaff)
normalized_input_oriented = normalized_input.replace(".mgz", "_oriented.mgz")
nibabel.save(normnii, normalized_input_oriented)


myreg = nibabel.load(currentstate).get_fdata()
mydiff = np.abs(myreg-norm_input)
# view([norm_input, myreg, mydiff])

mydiffnii = nibabel.Nifti1Image(mydiff, inaff)
diffpath =  currentstate.replace(".mgz", "_diff.mgz")
nibabel.save(mydiffnii, diffpath)


freesruferimage = p + "mri2fem-dataset/freesurfer/abby/cvs/final_CVSmorphed_toernie_norm.mgz" 
normimage = p + "mri2fem-dataset/freesurfer/ernie/mri/norm.mgz"

nom =nibabel.load(normimage).get_fdata()
fsreg = nibabel.load(freesruferimage).get_fdata()
freesurfer_difftonorm = np.abs(nom-fsreg)

fsdiffnii = nibabel.Nifti1Image(freesurfer_difftonorm, nibabel.load(normimage).affine)
diffpath2 =  freesruferimage.replace(".mgz", "_diff.mgz")
nibabel.save(fsdiffnii, diffpath2)


command += " "
command += normalized_input_oriented
command += " "
command += currentstate
command += " --colormap Heat -v " + diffpath + " --colormap Grayscale "
command += " "
command += normimage
command += " "
command += freesruferimage
command += " --colormap Heat -v " + diffpath2
os.system(command)