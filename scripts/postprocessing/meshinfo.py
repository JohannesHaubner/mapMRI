import os
import pathlib
from fenics import *

path = pathlib.Path("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/transported-meshes/abby-aqueduct/")
inputmesh = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mri2fem-dataset/meshes/reg-aqueduct/abby/affreg-ventricle-aq-boundarymesh.xml"

print("Inputmesh:")
bm = Mesh(inputmesh)

print(MeshQuality.radius_ratio_min_max(bm))

print(bm.hmin(), bm.hmax())

for subfolder in os.listdir(path):
    subfolder = path / subfolder

    meshfiles = [subfolder / x for x in os.listdir(subfolder) if str(x).endswith(".xml")]

    print(len(meshfiles))
    
    print(meshfiles[0])
    bm = Mesh(str(meshfiles[0]))

    print(MeshQuality.radius_ratio_min_max(bm))

    print(bm.hmin(), bm.hmax())