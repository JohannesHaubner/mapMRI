import numpy
import nibabel
import os
import SVMTK as svm

# # Abby
# outputdir="./data/meshes/abby/affine_registered_ventricles/"

# Ernie
outputdir="./data/meshes/ernie/ventricles/"

input=outputdir + "ventricles.stl"
output= outputdir + "ventricles"

res = 32
meshfile = output + ".mesh"

if not os.path.isfile(meshfile):
    surf = svm.Surface(input)
    domain = svm.Domain([surf])

    domain.create_mesh(res)
    domain.save(meshfile)


os.system("meshio-convert " + meshfile + " " + meshfile.replace(".mesh", ".xml"))
os.system("meshio-convert " + meshfile + " " + meshfile.replace(".mesh", ".xdmf"))

