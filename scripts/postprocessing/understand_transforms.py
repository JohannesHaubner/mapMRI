import nibabel
import numpy as np
import os
import matplotlib.pyplot as plt
from dgregister.helpers import read_vox2vox_from_lta
import pathlib
from nibabel.affines import apply_affine
from IPython import embed

dp = pathlib.Path("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/normalized")

regfolder = "registered"

regmatrix = read_vox2vox_from_lta(lta=dp / regfolder / "abbytoernie.lta")

abbytoernie = nibabel.load(str(dp / regfolder / "abbytoernie.mgz"))


abby = nibabel.load(str(dp / "input" / "abby" / "abby_brain.mgz"))
ernie = nibabel.load(str(dp / "input" / "ernie" / "ernie_brain.mgz"))
a = np.array(range(256))

i_coords, j_coords, k_coords = np.meshgrid(a,a,a, indexing='ij')
arr = np.array([i_coords, j_coords, k_coords])

arr = np.swapaxes(arr, 0, 1)
arr = np.swapaxes(arr, 1, 2)
arr = np.swapaxes(arr, 2, 3)
arr = arr.reshape((256**3,3))
# 

regvoxels = apply_affine(aff=(regmatrix), pts=arr)
regvoxels = np.where(regvoxels < 0, 0, regvoxels)
regvoxels = np.where(regvoxels > 255, 255, regvoxels)
regvoxels = regvoxels.astype(int)

regvoxels2 = apply_affine(aff=np.linalg.inv(regmatrix), pts=arr)
regvoxels2 = np.where(regvoxels2 < 0, 0, regvoxels2)
regvoxels2 = np.where(regvoxels2 > 255, 255, regvoxels2)
regvoxels2 = regvoxels2.astype(int)


a1= abby.get_fdata()[arr[:,0], arr[:,1], arr[:,2]]

a2 = abby.get_fdata()[regvoxels[:, 0], regvoxels[:,1], regvoxels[:,2]]
a3 = abby.get_fdata()[regvoxels2[:, 0], regvoxels2[:,1], regvoxels2[:,2]]

a3 = a3.reshape((256, 256, 256))
a2 = a2.reshape((256, 256, 256))
a1 = a1.reshape((256, 256, 256))

print("a1-abby", np.mean(np.abs(a1-abby.get_fdata())))
print("a2-abbytoernie", np.mean(np.abs(a2-abbytoernie.get_fdata())))
print("a3-abbytoernie", np.mean(np.abs(a3-abbytoernie.get_fdata())))


fig, ax = plt.subplots(2,2)
ax = ax.flatten()

idx = 100
axis = 2

names = ["abby", "abbytoernie", "a2", "a3"]
images = [abby, abbytoernie, a1, a3]

for a, name, img in zip(ax, names, images):
    # print(a, name, img)
    # breakpoint()
    # print(type(a), type(name), type(img))
    if not isinstance(img, np.ndarray):
        img = img.get_fdata()
    a.imshow(np.take(img, idx, axis), cmap="Greys_r", vmax=100)
    a.set_title(name)

plt.show()


embed()
exit()


fig, ax = plt.subplots(1, 3)
ax = ax.flatten()

idx = 100
axis = 0

names = ["abby", "abbytoernie", "ernie"]
images = [abby, abbytoernie, ernie,]

for a, name, img in zip(ax, names, images):
    # print(a, name, img)
    # breakpoint()
    # print(type(a), type(name), type(img))
    a.imshow(np.take(img.get_fdata(), idx, axis), cmap="Greys_r", vmax=100)


plt.show()