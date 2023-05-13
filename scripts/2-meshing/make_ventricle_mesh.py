import numpy
import nibabel
import os
import SVMTK as svm

# Abby
outputdir="./data/meshes/reg-aqueduct/abby/"

# Ernie
outputdir="./data/meshes/aqueduct/ernie/"

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

