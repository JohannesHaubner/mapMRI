"""
Deform a mesh using a sequence of velocity fields-based mappings
"""
import os
import argparse
import json
import numpy as np
from fenics import *
from fenics_adjoint import *
import meshio
import pathlib
import nibabel
from dgregister.helpers import get_bounding_box_limits


def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        print(*args)
    else:
        pass




class Meshdata():
    """
    Convenience wrapper containing some files and parameters needed to transform the mesh.
    """

    def __init__(self, input_meshfile) -> None:


        self.box = np.load("./data/normalized/cropped/box.npy")
        self.pad = 2

        self.affine = nibabel.load("./data/normalized/registered/abbytoernie.mgz").affine

        self.registration_lta = "./data/" + "normalized/registered/abbytoernie.lta"
        
        self.input_meshfile = input_meshfile

        self.original_input = "./data/normalized/registered/abbytoernie.mgz"
        self.original_target = "./data/freesurfer/ernie/mri/brain.mgz"

        print_overloaded("Read meshfile", self.input_meshfile)
            
        self.vox2ras_input = nibabel.load(self.original_input).header.get_vox2ras_tkr()
        self.vox2ras_target = nibabel.load(self.original_target).header.get_vox2ras_tkr()

        self.inputmesh = Mesh(self.input_meshfile)

        print("Mesh.coordinates().shape=", self.inputmesh.coordinates().shape)

        bounds = get_bounding_box_limits(self.box)
        self.dxyz = [bounds[x].start for x in range(3)]

    def meshcopy(self) -> Mesh:
        return Mesh(self.input_meshfile)
   







parser = argparse.ArgumentParser()
parser.add_argument("--folders", nargs="+", type=str, default=[])
parser.add_argument("--input_meshfile", required=True, type=str, help="Input mesh ")
parser.add_argument("--meshoutputfolder", required=True, type=str, help="Folder where all the transported meshes should be stored.")
parser.add_argument("--remesh", action="store_true", default=False, help="Remesh and fix surface mesh using SVMTK")
parserargs = vars(parser.parse_args())

deformation_hyperparameters = json.load(open(parserargs["folders"][0] + "hyperparameters.json"))

from dgregister.meshtransform import map_mesh

parserargs["folders"] = sorted(parserargs["folders"])

mapfiles = {}
hyperparameters = {}
outputfolders = {}

for idx, folder in enumerate(parserargs["folders"]):

    deformation_hyperparameters = json.load(open(folder + "hyperparameters.json"))

    if idx == 0 and "slurmid" in deformation_hyperparameters.keys():
        previous_slurmid = deformation_hyperparameters["slurmid"]
    elif "slurmid" in deformation_hyperparameters.keys():
        assert str(previous_slurmid) in deformation_hyperparameters["starting_state"]
        previous_slurmid = deformation_hyperparameters["slurmid"]

    print_overloaded("Folder:        ", folder)
    print_overloaded("Starting guess:", deformation_hyperparameters["starting_state"])

    assert deformation_hyperparameters["starting_guess"] is None
        
    hyperparameters[folder] = deformation_hyperparameters

    mapfiles[folder] = folder + "all" + ".hdf"

nx = hyperparameters[folder]["target.shape"][0]
ny = hyperparameters[folder]["target.shape"][1]
nz = hyperparameters[folder]["target.shape"][2]


state_space, state_degree = hyperparameters[folder]["state_functionspace"], hyperparameters[folder]["state_functiondegree"]

cubemesh = BoxMesh(MPI.comm_world, Point(0.0, 0.0, 0.0), Point(nx, ny, nz), nx, ny, nz)
V3 = VectorFunctionSpace(cubemesh, hyperparameters[folder]["velocity_functionspace"], hyperparameters[folder]["velocity_functiondegree"],)

mappings = []


for idx, folder in enumerate(parserargs["folders"]):

    print(idx, folder)    

    hdf = HDF5File(cubemesh.mpi_comm(), mapfiles[folder], "r")
    mapping = Function(V3)
    hdf.read(mapping, "coordinatemapping")
    hdf.close()

    assert norm(mapping) != 0
    print_overloaded("Read mapping")

    
    mappings.append(mapping)




data = Meshdata(input_meshfile=parserargs["input_meshfile"])

if not parserargs["meshoutputfolder"].endswith("/"):
    parserargs["meshoutputfolder"] += "/"

os.makedirs(parserargs["meshoutputfolder"], exist_ok=True)

targetmesh1 = map_mesh(mappings=mappings, data=data, remesh=parserargs["remesh"], exportdir=parserargs["meshoutputfolder"])


print("map_mesh called succesfully, now storing meshes")

meshes = {}
meshes["transformed_mesh"] = targetmesh1

for meshname, meshobject in meshes.items():

    xmlfile = parserargs["meshoutputfolder"] + meshname + ".xml"

    File(xmlfile) << meshobject

    transormed_xmlmesh = meshio.read(xmlfile)
    transormed_xmlmesh.write(xmlfile.replace(".xml", ".xdmf"))
    

    # Store as hdf File for use in further FEniCS simulation
    hdf = HDF5File(meshobject.mpi_comm(), xmlfile.replace(".xml", ".hdf"), "w")
    hdf.write(meshobject, "mesh")
    hdf.close()

    if meshobject.topology().dim() != 2:
        bmesh = BoundaryMesh(meshobject, "exterior")
        boundarymeshfile = xmlfile.replace(".xml", "_boundary.xml")
        File(boundarymeshfile) << bmesh

        transormed_boundary_xmlmesh = meshio.read(boundarymeshfile)
        transormed_boundary_xmlmesh.write(boundarymeshfile.replace(".xml", ".xdmf"))
        transormed_boundary_xmlmesh.write(boundarymeshfile.replace(".xml", ".stl"))
    else:
        transormed_xmlmesh.write(xmlfile.replace(".xml", ".stl"))

    print("Stored ", meshname, "in all formats")

if parserargs["meshoutputfolder"] is not None:

    with open(pathlib.Path(parserargs["meshoutputfolder"]) / "parserargs.json", 'w') as outfile:
        json.dump(parserargs, outfile, sort_keys=True, indent=4)

    print("Stored parserargs")

print("Stored meshes, script finished.")