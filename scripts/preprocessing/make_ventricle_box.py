from fenics import *
import h5py
from IPython import embed
import numpy as np
from parse import parse
import nibabel
import os
import matplotlib.pyplot as plt
from nibabel.affines import apply_affine
from dgregister.helpers import get_larget_box, get_bounding_box_limits, cut_to_box, pad_with

meshpath = "/home/basti/Dropbox (UiO)/068meshes/brain_cp/brain_mesh_refined.h5"

oimage_path = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/hydrocephalus/freesurfer/021/mri/brain.mgz"
oimage = nibabel.load(oimage_path)

reg_image_path = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/hydrocephalus/normalized/registered/021to068.mgz"
reg_image = nibabel.load(reg_image_path)

target_image_path = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/hydrocephalus/freesurfer/068/mri/brain.mgz"
target_image_aseg = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/hydrocephalus/freesurfer/068/mri/aseg.mgz"
target_image = nibabel.load(target_image_path)

# image2 = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/hydrocephalus/normalized/nyul_normalized/021to068_nyul.mgz"
# image2 = nibabel.load(image2)

# fig, ax = plt.subplots(2,1)
# ax = ax.flatten()

# idx = 100
# axis = 1

# names = ["registered", "normalized"]
# images = [image, image2]

# for a, name, img in zip(ax, names, images):
#     # print(a, name, img)
#     # breakpoint()
#     # print(type(a), type(name), type(img))
#     a.imshow(np.take(img.get_fdata(), idx, axis), cmap="Greys_r", vmax=100)

# for a, name, img in zip(ax, names, images):
#     plt.figure()
#     plt.hist(img.get_fdata().flatten())
#     plt.yscale("log")

# plt.show()

lta = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/hydrocephalus/normalized/registered/021to068.lta"

"""
type      = 0 # LINEAR_VOX_TO_VOX
nxforms   = 1
mean      = 128.0000 112.0000 115.0000
sigma     = 10000.0000
1 4 4
9.998921751976013e-01 -1.197870355099440e-02 -8.493579924106598e-03 2.367060422897339e+00
1.149188354611397e-02 9.984082579612732e-01 -5.521716922521591e-02 8.640504837036133e+00
9.141489863395691e-03 5.511360615491867e-02 9.984382390975952e-01 -1.008353900909424e+01
0.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00 1.000000000000000e+00
src volume info
"""

File = open(lta)

lines = File.readlines()

regmatrix_v2v = []

for line in lines:
    # print(line)

    res = parse("{} {} {} {}", line.replace("\n", ""))

    try:
        a, b, c, d = float(res[0]), float(res[1]), float(res[2]), float(res[3])
        print(a,b,c,d)
        print("*"*80)

        regmatrix_v2v.append([a,b,c,d])
    except:
        pass


File.close()

# embed()


regmatrix_v2v = np.array(regmatrix_v2v)


# f = h5py.File(meshpath, 'r')
# print(list(f.keys()))
# exit()

domainmesh = Mesh()

hdf = HDF5File(domainmesh.mpi_comm(), meshpath, "r")
hdf.read(domainmesh, "mesh", False)
# V = FunctionSpace(mesh, "DG", 1)
# u = Function(V)
# hdf.read(u, "Img")
subdomains = MeshFunction("size_t", domainmesh, domainmesh.topology().dim())
hdf.read(subdomains, "/subdomains")

hdf.close()

xyz = domainmesh.coordinates()

aq = subdomains.where_equal(4)
ventricles = subdomains.where_equal(6)

cells = domainmesh.cells()


aqcells = cells[aq, :]
aqxyz= xyz[aqcells.flatten(), :]

ventriclescells = cells[ventricles, :]
ventriclesxyz= xyz[ventriclescells.flatten(), :]

csfxyz = np.vstack([aqxyz, ventriclesxyz])

lower = np.min(csfxyz, axis=0)
upper = np.max(csfxyz, axis=0)


vox2ras = reg_image.header.get_vox2ras_tkr()
ras2vox = np.linalg.inv(vox2ras)


os.makedirs(oimage_path.replace("mri/brain.mgz", "testouts"), exist_ok=True)

command = "freeview " + oimage_path + " " + reg_image_path + " "
matrices = [np.eye(4), regmatrix_v2v]
filenames =["ventricularmask", "reg_ventricularmask"]
images= [oimage, reg_image]

for matrix, filename, image in zip(matrices, filenames, images):

    # ijk1 = apply_affine(aff, ijk1)

    # np.matmul(regmatrix_v2v, aff)

    ijk = apply_affine(ras2vox, csfxyz)

    ijk = apply_affine(matrix, ijk)

    ijk = np.round(ijk, 0)

    # ijk = np.unique(ijk, axis=0)
    ijk = ijk.astype(int)

    mask = np.zeros_like(reg_image.get_fdata())
    box = np.zeros_like(reg_image.get_fdata())

    mask[ijk[:,0], ijk[:,1], ijk[:,2]] = 1

    box[np.min(ijk[:,0])-2:np.max(ijk[:,0])+2,np.min(ijk[:,1])-2:np.max(ijk[:,1])+2,np.min(ijk[:,2])-2:np.max(ijk[:,2])+2] = 1

    

    outfile = oimage_path.replace("mri/brain.mgz", "testouts/" + filename + ".mgz")
    nii = nibabel.Nifti1Image(mask.astype(float), image.affine)
    nibabel.save(nii, outfile)
    command += "--colormap Heat -v " + outfile + " "

    outfile = oimage_path.replace("mri/brain.mgz", "testouts/" + filename + "_box" + ".mgz")
    nii = nibabel.Nifti1Image(box.astype(float), image.affine)
    nibabel.save(nii, outfile)

    command += "--colormap Heat -v " + outfile + " "

print(command)

imagepath = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/hydrocephalus/normalized/nyul_normalized/"

box1 = nibabel.load(oimage_path.replace("mri/brain.mgz", "testouts/" + "reg_ventricularmask" + "_box" + ".mgz")).get_fdata()

csf_labels = [4, 5, 14, 15, 24, 43, 44]

aseg = nibabel.load(target_image_aseg)
affine = aseg.affine

aseg = aseg.get_fdata().astype(int)

csf_mask = np.zeros(tuple(aseg.shape), dtype=bool)

for csf_label in csf_labels:
    csf_mask += (aseg == csf_label) 

# box = np.where(box1 | csf_mask, True, False)


box = get_larget_box([box1, csf_mask])
np.save(oimage_path.replace("mri/brain.mgz", "testouts/" + "box_all" + ".npy"), box)

outfile = oimage_path.replace("mri/brain.mgz", "testouts/" + "box_all" + ".mgz")
nii = nibabel.Nifti1Image(box.astype(float), affine)
nibabel.save(nii, outfile)

print(box.shape)

limits = get_bounding_box_limits(box)

limits2 = []
for l in limits:
    limits2.append(slice(l.start -2, l.stop + 2, None))

limits = limits2

outpath = imagepath.replace("normalized/nyul_normalized/", "ventricles/")

os.makedirs(outpath, exist_ok=True)

for file in os.listdir(imagepath):

    image = nibabel.load(imagepath + file).get_fdata()


    cropped = cut_to_box(image, box_bounds=limits, inverse=False, cropped_image=None)

    cropped = np.pad(cropped, 2, pad_with)

    nii = nibabel.Nifti1Image(cropped.astype(float), np.eye(4))
    nibabel.save(nii, outpath + file.replace(".mgz", "_ventriclemasked.mgz"))


# embed()
# exit()
