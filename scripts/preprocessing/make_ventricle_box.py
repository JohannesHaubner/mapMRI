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
from dgregister.helpers import read_vox2vox_from_lta
import meshio
from tqdm import tqdm
meshpath = "/home/basti/Dropbox (UiO)/068meshes/brain_cp/brain_mesh_refined.h5"
# boundarymesh.shape (24742, 3)

meshpath = "/home/basti/Dropbox (UiO)/068meshes/brain_cp/brain_mesh.h5"
# boundarymesh.shape (23819, 3)


oimage_path = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/hydrocephalus/freesurfer/021/mri/brain.mgz"
oimage = nibabel.load(oimage_path)

meshoutput_path = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/hydrocephalus/meshes/"
os.makedirs(meshoutput_path, exist_ok=True)
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

regmatrix_v2v = read_vox2vox_from_lta(lta)


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

# breakpoint()

csfcells = np.vstack([aqcells, ventriclescells])
csfxyz = np.vstack([aqxyz, ventriclesxyz])


cell_indices = np.unique(csfcells.flatten())
new_indices = np.array(range(len(cell_indices)))

lookup = {}
inverse_lookup = {}

xyz2 = np.zeros((cell_indices.size, 3))
# Full mesh index -> index in [0, N] for cells in CSF
for i, cellidx in enumerate(cell_indices):
    lookup[cellidx] = i    
    xyz2[i, :] = xyz[cellidx, :]

newcells = np.zeros_like(csfcells)



progress = tqdm(total=csfcells.shape[0])

for cell_idx in range(csfcells.shape[0]):

    for i in range(4):
        
        newcells[cell_idx, i] = lookup[csfcells[cell_idx, i]]
        
    progress.update(1)


csfcells = newcells
xyz = xyz2

mesh = meshio.Mesh(
    xyz,
    [("tetra", csfcells)],
)

print(np.max(xyz, axis=0)-np.min(xyz, axis=0))

mesh.write(meshoutput_path + "ventricles.xdmf")

os.system("meshio-convert " + meshoutput_path + "ventricles.xdmf" + " " + meshoutput_path + "ventricles.xml")



ventricle_mesh = Mesh(meshoutput_path + "ventricles.xml")

boundarymesh = BoundaryMesh(ventricle_mesh, "exterior")

boundarymeshfile = meshoutput_path + "ventricle_boundary.xml"
mesh_file = File(boundarymeshfile)
mesh_file << boundarymesh

print("boundarymesh.shape", boundarymesh.coordinates().shape)

os.system("meshio-convert " + boundarymeshfile + " " + boundarymeshfile.replace(".xml", ".xdmf"))

print("paraview ", boundarymeshfile.replace(".xml", ".xdmf"))
exit()

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
