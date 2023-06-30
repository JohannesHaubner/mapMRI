import os
import numpy as np
import nibabel


def tukey(x,c=255):

    return np.mean(np.where(np.abs(x) <= c, c ** 2 / 2 * (1 - (1 - x ** 2 / c ** 2) ** 3), c ** 2 / 2))

# inputimage = "./data/freesurfer/ernie/mri/brain.mgz"

normalized_input = "./data/normalized/nyul_normalized/ernie_brain_nyul.mgz"
inputimage = "./data/freesurfer/ernie/mri/norm.mgz"

currentstate = "./outputs/my_registration_3/norm_registered/CurrentState.mgz"
currentstate2 = "./outputs/my_registration_4/norm_registered/CurrentState.mgz"

# currentstate = "./outputs/my_registration_3/CurrentState.mgz"
# currentstate2 = "./outputs/my_registration_4/CurrentState.mgz"

inputimage = nibabel.load(inputimage)
inaff = inputimage.affine
inputimage = inputimage.get_fdata()

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

ernie_norm =nibabel.load(normimage).get_fdata()
fsreg = nibabel.load(freesruferimage).get_fdata()
freesurfer_difftonorm = np.abs(ernie_norm-fsreg)

fsdiffnii = nibabel.Nifti1Image(freesurfer_difftonorm, nibabel.load(normimage).affine)
fsdiffpath =  freesruferimage.replace(".mgz", "_diff.mgz")
nibabel.save(fsdiffnii, fsdiffpath)




print("number of outliers", np.where(np.abs(myreg2) > 1e3, 1,0).sum(), "(voxels with value > 1000)")
# exit()

import matplotlib.pyplot as plt
plt.figure()
plt.title("Distribution of absolute voxel intensities,\n 4 velocity transform")
plt.hist(np.abs(myreg2).flatten())
plt.yscale("log")

plt.figure()
plt.title("Distribution of absolute voxel intensities,\n 3 velocity transform")
plt.hist(np.abs(myreg).flatten())
plt.yscale("log")

plt.figure()
plt.title("Distribution of absolute voxel intensities, \n target image")
plt.hist(np.abs(ernie_norm).flatten())
plt.yscale("log")


# plt.show()

# exit()
erniemax = ernie_norm.max()
print("Max in ernie norm", erniemax)

print("l2-difference between target and 3-velocity transform")
print(np.mean((ernie_norm-myreg)**2))

print("l2-difference between target and 4-velocity transform")
print(np.mean((ernie_norm-myreg2)**2))

print("l2-difference between target and FreeSurfer cvs-transform")
print(np.mean((ernie_norm-fsreg)**2))


print("Tukey difference between target and 3-velocity transform")
print(tukey(ernie_norm-myreg, c=erniemax))

print("Tukey  difference between target and 4-velocity transform")
print(tukey(ernie_norm-myreg2, c=erniemax))

print("Tukey  difference between target and FreeSurfer cvs-transform")
print(tukey(ernie_norm-fsreg, c=erniemax))

print("Tukey difference between target and 3-velocity transform for c=50")
print(tukey(ernie_norm-myreg, c=50))

print("Tukey  difference between target and 4-velocity transform for c=50")
print(tukey(ernie_norm-myreg2, c=50))

print("Tukey  difference between target and FreeSurfer cvs-transform for c=50")
print(tukey(ernie_norm-fsreg, c=50))

print("Tukey difference between target and 3-velocity transform for c=110")
print(tukey(ernie_norm-myreg, c=110))

print("Tukey  difference between target and 4-velocity transform for c=110")
print(tukey(ernie_norm-myreg2, c=110))

print("Tukey  difference between target and FreeSurfer cvs-transform for c=110")
print(tukey(ernie_norm-fsreg, c=110))


# exit()

command = "freeview --slice 125 110 111 --viewport axial"
# command += " "
# command += normalized_input_oriented
# command += " "
# command += currentstate
command += " --colormap Heat:heatscale=10,30,50 -v " + diffpath
command += " "
command += " --colormap Heat:heatscale=10,30,50 -v " + diffpath2 + " --colormap Grayscale "
command += " "
command += normimage
command += " "
# command += freesruferimage
command += " --colormap Heat:heatscale=10,30,50 -v " + fsdiffpath
os.system(command)