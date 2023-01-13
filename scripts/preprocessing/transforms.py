import nibabel
import numpy as np
from nibabel.affines import apply_affine
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



abby = nibabel.load("abby_brain.mgz")
ernie = nibabel.load("ernie_brain.mgz")
abby2ernie = nibabel.load("abbytoernie.mgz")

images = {"abby": abby, "abby2ernie": abby2ernie, "ernie": ernie, }

# plt.figure()
# plt.title("abby")
# plt.imshow(abby.get_fdata()[100, ...], cmap="Greys_r")
# plt.figure()
# plt.title("abby2ernie")
# plt.imshow(abby2ernie.get_fdata()[100, ...], cmap="Greys_r")
# plt.colorbar()

# plt.figure()
# plt.title("abby-abby2ernie")
# plt.imshow(abby.get_fdata()[100, ...]-abby2ernie.get_fdata()[100, ...], cmap="jet")
# plt.colorbar()


# plt.figure()
# plt.title("ernie")
# plt.imshow(ernie.get_fdata()[100, ...], cmap="Greys_r")
# plt.show()



for name, image in images.items():
    print(name)
    print(np.round(image.affine))
    print(np.round(image.header.get_vox2ras()))


ras2vox_a = images["abby"].header.get_vox2ras()
vox2ras_a= np.linalg.inv(ras2vox_a)

ras2vox_e = images["ernie"].header.get_vox2ras()



X = np.linspace(0, 255, 256)
x,y,z=np.meshgrid(X, X, X)

index = np.stack((x, y, z), axis=-1)

# pqr = apply_affine(ras2vox_e, apply_affine(vox2ras_a, index))


aff = np.array([[9.800471663475037e-01, -5.472707748413086e-02, 1.910823285579681e-01, -1.452283763885498e+01],
                [4.260246828198433e-02, 9.968432784080505e-01, 6.699670851230621e-02, -1.174131584167480e+01],
                [-1.941456645727158e-01, -5.751936137676239e-02, 9.792849421501160e-01, 3.610760116577148e+01],
                [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.000000000000000e+00]
                ])

pqr = apply_affine(aff, index)


pqr = np.where(pqr < 0, 0, pqr)
pqr = np.where(pqr > 255, 255, pqr)


# np.save("pqr.npy", pqr)

pqr = np.rint(pqr).astype(int)

# indeximg = nibabel.Nifti1Image(index, abby2ernie.affine)
# nibabel.save(indeximg, "indices.mgz")

index=index.astype(int)

# subprocess.run("mri_vol2vol --reg abbytoernie.lta --mov indices.mgz --o index_inverse.mgz --targ abby_brain.mgz", shell=True)

# index_inverse = nibabel.load("index_inverse.mgz").get_fdata().astype(int)

abby_arr = abby.get_fdata()

abby_pqr = np.zeros_like(abby_arr)



index1 = pqr
index2 = index

# index1 = index
# index2 = index_inverse


abby2erniearr = abby2ernie.get_fdata()

abby_pqr[index1[..., 0], index1[..., 1], index1[..., 2]] = abby_arr[index2[..., 0], index2[..., 1], index2[..., 2]]


imgz = nibabel.Nifti1Image(abby_pqr, abby.affine)
nibabel.save(imgz, "abbyqr.mgz")

print("diff between Abby_pqr and Abby      ", np.sum(np.abs(abby_pqr - abby_arr)) / np.sum(np.abs(abby_arr)))
print("diff between Abby_pqr and Abby2ernie", np.sum(np.abs(abby_pqr - abby2erniearr) / np.sum(np.abs(abby_arr))), "should be small ?")

img1=abby_pqr
img2=abby_arr # erniearr
img3=abby2erniearr

ax, sl = 1, 100



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.imshow(np.take(img1, indices=sl, axis=ax), cmap="Greys_r")
ax2.imshow(np.take(img2, indices=sl, axis=ax), cmap="Greys_r")

im3 = ax3.imshow(np.take(img1-img2, indices=sl, axis=ax), cmap="jet")
divider = make_axes_locatable(ax3)
ax3.set_title("abby-abby_pqr")
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation='vertical')

im4 = ax4.imshow(np.take(img1-img3, indices=sl, axis=ax), cmap="jet")
divider = make_axes_locatable(ax4)
ax4.set_title("abby2ernie-abby_pqr")
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im4, cax=cax, orientation='vertical')

plt.show()

