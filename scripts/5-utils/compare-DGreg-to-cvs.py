import os
import numpy as np
import nibabel

inputimage = "./data/freesurfer/ernie/mri/brain.mgz"

normalized_input = "./data/normalized/nyul_normalized/ernie_brain_nyul.mgz"
currentstate = "./registration/normalized-outputs/447918/RK100A0.01LBFGS150/CurrentState.mgz"
currentstate2 = "./registration/normalized-outputs/450276/RK100A0.01LBFGS150/CurrentState.mgz"

inaff = nibabel.load(inputimage).affine

norm_input = nibabel.load(normalized_input).get_fdata()

normnii = nibabel.Nifti1Image(norm_input, inaff)
normalized_input_oriented = normalized_input.replace(".mgz", "_oriented.mgz")
nibabel.save(normnii, normalized_input_oriented)

diffaffine = nibabel.load("./data/normalized/registered/abbytoernie.mgz").affine

inaff= diffaffine

myreg = nibabel.load(currentstate).get_fdata()
mydiff = np.abs(myreg-norm_input)
# view([norm_input, myreg, mydiff])

mydiffnii = nibabel.Nifti1Image(mydiff, inaff)
diffpath =  currentstate.replace(".mgz", "_diff.mgz")
nibabel.save(mydiffnii, diffpath)


myreg2 = nibabel.load(currentstate2).get_fdata()
mydiff2 = np.abs(myreg2-norm_input)
# view([norm_input, myreg, mydiff])

mydiffnii2 = nibabel.Nifti1Image(mydiff2, inaff)
diffpath2 =  currentstate2.replace(".mgz", "_diff.mgz")
nibabel.save(mydiffnii2, diffpath2)



freesruferimage = "./data/freesurfer/abby/cvs/final_CVSmorphed_toernie_norm.mgz" 
normimage = "./data/freesurfer/ernie/mri/norm.mgz"

nom =nibabel.load(normimage).get_fdata()
fsreg = nibabel.load(freesruferimage).get_fdata()
freesurfer_difftonorm = np.abs(nom-fsreg)

fsdiffnii = nibabel.Nifti1Image(freesurfer_difftonorm, nibabel.load(normimage).affine)
fsdiffpath =  freesruferimage.replace(".mgz", "_diff.mgz")
nibabel.save(fsdiffnii, fsdiffpath)

command = "freeview"
command += " "
command += normalized_input_oriented
command += " "
command += currentstate
command += " --colormap Heat -v " + diffpath
command += " "
command += " --colormap Heat -v " + diffpath2 + " --colormap Grayscale "
command += " "
command += normimage
command += " "
command += freesruferimage
command += " --colormap Heat -v " + fsdiffpath
os.system(command)