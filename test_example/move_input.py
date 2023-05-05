import nibabel
import numpy as np
import os
from fenics import *
import meshio
import dgregister.config as config

roiname = "input"
hyperparameters = {"optimize": False, "state_functiondegree": 1, "state_functionspace":"DG", "input": roiname + ".mgz", "timestepping":"explicitEuler"}
hyperparameters["preconditioner"] = "amg"
hyperparameters["solver"] = "krylov"

config.hyperparameters = hyperparameters


from dgregister.meshtransform import make_mapping, map_mesh


os.chdir("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d")

path = "./"


cubemesh = Mesh()

# VelocitFile = "NewTarget.hdf"
# readname = "velocity"
# outname = "transformed_input_mesh"

VelocitFile = "VelocityField.hdf"
readname = "-1"
outname = "transformed_input_mesh_optimized"


hdf = HDF5File(cubemesh.mpi_comm(), VelocitFile, "r")
hdf.read(cubemesh, "mesh", False)

vCG = VectorFunctionSpace(cubemesh, "CG", 1)
v = Function(vCG)

hdf.read(v, readname)


hdf.close()

hyperparameters["smoothen"] = False

from make_target import MaxIter, DeltaT

hyperparameters["max_timesteps"] = MaxIter
hyperparameters["DeltaT"] = DeltaT
hyperparameters["MassConservation"] = False
hyperparameters["DGtransport"] = True

hyperparameters["inverseRAS"] = False

mapfile = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/Mapping.hdf"

os.system("rm " + mapfile)

if os.path.isfile(mapfile):
    hdf = HDF5File(cubemesh.mpi_comm(), mapfile, "r")
    mapping = Function(vCG)
    velocity = Function(vCG)
    hdf.read(mapping, "mapping")
    hdf.close()
else:

    mapping = make_mapping(cubemesh, velocity=v, 
                    hyperparameters=hyperparameters, ocd=False, dgtransport=hyperparameters["DGtransport"])

    hdf = HDF5File(cubemesh.mpi_comm(), mapfile, "w")
    hdf.write(mapping, "mapping")
    hdf.close()

File("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/mapping.pvd") << mapping


xmlfile1 = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/input8.xml"
imgfile1 = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/input.mgz"
imgfile2 = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/newtarget.mgz"


brainmesh2 = map_mesh(xmlfile1, imgfile1, imgfile2, mapping, box=None, 
                    registration_affine=None,
                    invert_registration_affine=hyperparameters["inverseRAS"],
                    outfolder="/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/",
                    npad=0, raise_errors=True,
                    coarsening_factor=1)

File("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/velocity.pvd") << v

# # Store as xdmf file for paraview visualization
xmlfile3 =  "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/" + outname + ".xml"
# with XDMFFile(xmlfile3) as xdmf:
#     xdmf.write(brainmesh2) # , "mesh")

File(xmlfile3) << brainmesh2


transormed_xmlmesh = meshio.read(xmlfile3)
transormed_xmlmesh.write(xmlfile3.replace(".xml", ".xdmf"))

# Store as hdf File for use in further FEniCS simulation
hdf = HDF5File(brainmesh2.mpi_comm(), xmlfile3.replace(".xml", ".hdf"), "w")
hdf.write(brainmesh2, "mesh")
hdf.close()

inputmesh = Mesh(xmlfile1)


for m, mname in zip([inputmesh, brainmesh2],["Input mesh", "After transform"]):
    print(mname)
    print("Ratio min max", MeshQuality.radius_ratio_min_max(m))
    print("hmax",  m.hmax())
    print("hmin",  m.hmin())