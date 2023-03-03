import numpy
import nibabel
import os
import SVMTK as svm

input="/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/meshes/ventricles/abby/ventricles.stl"
output="/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/meshes/ventricles/abby/ventricles"

res = 16
meshfile = output + ".mesh"

if not os.path.isfile(meshfile):
    surf = svm.Surface(input)
    domain = svm.Domain([surf])

    domain.create_mesh(res)
    domain.save(meshfile)


os.system("meshio-convert " + meshfile + " " + meshfile.replace(".mesh", ".xml"))
os.system("meshio-convert " + meshfile + " " + meshfile.replace(".mesh", ".xdmf"))

