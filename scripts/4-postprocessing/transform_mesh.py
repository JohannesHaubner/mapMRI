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
parser = argparse.ArgumentParser()

parser.add_argument("--mapping_only", help="only create the coordinate mapping and exit (no mesh transform)", action="store_true", default=False)
parser.add_argument("--folders", nargs="+", type=str, default=[])
parser.add_argument("--recompute_mapping", action="store_true", default=False)
parser.add_argument("--outputfoldername", type=str, default="meshtransform/")
parser.add_argument("--meshoutputfolder", type=str, help="Folder where all the transported meshes should be stored.")
parser.add_argument("--remesh", action="store_true", default=False, help="Remesh")
parserargs = vars(parser.parse_args())

if len(parserargs["folders"]) > 1:
    raise NotImplementedError("Setting omega, epsilon for several deformations not yet implemented")
else:

    deformation_hyperparameter = json.load(open(parserargs["folders"][0] + "hyperparameters.json"))

import dgregister.config as config

config.EPSILON = deformation_hyperparameter["epsilon"]
config.OMEGA = deformation_hyperparameter["omega"]

from dgregister.meshtransform import map_mesh, make_mapping
from dgregister.helpers import get_lumped_mass_matrices


parserargs["folders"] = sorted(parserargs["folders"])


print_overloaded("*"*80)

for key, item in parserargs.items():
    print_overloaded(key, item)

print_overloaded("*"*80)

if not parserargs["outputfoldername"].endswith("/"):
    parserargs["outputfoldername"] += "/"

assert parserargs["outputfoldername"][0] != "/"


if parserargs["mapping_only"]:
    print_overloaded("--mapping_only is set, will only create the mapping and exit()")

mapfiles = {}
hyperparameters = {}
outputfolders = {}

previous_slurmid = None

for idx, folder in enumerate(parserargs["folders"]):
    

    deformation_hyperparameter = json.load(open(folder + "hyperparameters.json"))
    if idx == 0:
        if not parserargs["mapping_only"]:
            assert deformation_hyperparameter["starting_state"] is None
        previous_slurmid = deformation_hyperparameter["slurmid"]
    else:
        assert str(previous_slurmid) in deformation_hyperparameter["starting_state"]
        previous_slurmid = deformation_hyperparameter["slurmid"]

    print_overloaded("Folder:        ", folder)
    print_overloaded("Starting guess:", deformation_hyperparameter["starting_state"])

    assert deformation_hyperparameter["starting_guess"] is None
        
    hyperparameters[folder] = deformation_hyperparameter

    outputfolder = str(pathlib.Path(folder, parserargs["outputfoldername"]))

    if not outputfolder.endswith("/"):
        outputfolder += "/"

    outputfolders[folder] = outputfolder

    os.makedirs(outputfolder, exist_ok=True)

    mapfiles[folder] = outputfolder + "all" + ".hdf"



mappingname = "coordinatemapping"

nx = hyperparameters[folder]["target.shape"][0]
ny = hyperparameters[folder]["target.shape"][1]
nz = hyperparameters[folder]["target.shape"][2]


state_space, state_degree = hyperparameters[folder]["state_functionspace"], hyperparameters[folder]["state_functiondegree"]

cubemesh = BoxMesh(MPI.comm_world, Point(0.0, 0.0, 0.0), Point(nx, ny, nz), nx, ny, nz)

print_overloaded("cubemesh size", nx, ny, nz)

V3 = VectorFunctionSpace(cubemesh, hyperparameters[folder]["velocity_functionspace"], hyperparameters[folder]["velocity_functiondegree"],)
M_lumped_inv = None

mappings = []

# NOTE
# NOTE reversed()  can be used to apply the transformations in reverse order. 
# reverse should be False for mesh
parserargs["reverse"] = False

if not parserargs["reverse"]:
    folders=parserargs["folders"]
else:
    folders=reversed(parserargs["folders"])


for idx, folder in enumerate(folders):

    print(idx, folder)    

    if os.path.isfile(mapfiles[folder]) and (not parserargs["recompute_mapping"]):
        hdf = HDF5File(cubemesh.mpi_comm(), mapfiles[folder], "r")
        mapping = Function(V3)
        hdf.read(mapping, mappingname)
        hdf.close()

        assert norm(mapping) != 0
        print_overloaded("Read mapping")

    else:
        if M_lumped_inv is None:
            _, M_lumped_inv = get_lumped_mass_matrices(vCG=V3)

        l2_control = Function(V3)

        with XDMFFile(folder + "/Control_checkpoint.xdmf") as xdmf:
            xdmf.read_checkpoint(l2_control, "CurrentV")

        mapping = make_mapping(cubemesh, control=l2_control, M_lumped_inv=M_lumped_inv, hyperparameters=hyperparameters[folder])

        hdf = HDF5File(cubemesh.mpi_comm(), mapfiles[folder], "w")
        hdf.write(mapping, mappingname)
        hdf.close()
        print_overloaded("Created vector function that maps xyz mesh coordinates")

    mappings.append(mapping)

if parserargs["mapping_only"]:
    print_overloaded("--mapping_only is set, created the mapping. exit()")
    exit()



class Meshdata():
    """Convenience wrapper containing some files and parameters needed to transform the mesh.
    """

    def __init__(self, input, target) -> None:

        assert "abby" in input
        assert "ernie" in target
        box = np.load("./data/normalized/cropped/box.npy")
        space = 0
        pad = 2

        aff3 = nibabel.load("./data/normalized/registered/abbytoernie.mgz").affine

        self.registration_lta = "./data/" + "normalized/registered/abbytoernie.lta"
        
        self.input_meshfile = "./data/meshes/abby-lh-affineregistered.xml"
        self.original_input = "./data/normalized/registered/abbytoernie.mgz"
        self.original_target = "./data/normalized/input/ernie/" + "ernie_brain.mgz"

        print_overloaded("Read meshfile", self.input_meshfile)
            
        self.vox2ras_input = nibabel.load(self.original_input).header.get_vox2ras_tkr()
        self.vox2ras_target = nibabel.load(self.original_target).header.get_vox2ras_tkr()

        self.inputmesh = Mesh(self.input_meshfile)

        print("Mesh has", self.inputmesh.coordinates().shape, "shape")

        self.box = box
        self.space = space
        self.pad = pad
        self.affine = aff3

        bounds = get_bounding_box_limits(self.box)
        self.dxyz = [bounds[x].start for x in range(3)]

    def meshcopy(self) -> Mesh:
        return Mesh(self.input_meshfile)
    


data = Meshdata(hyperparameters[folder]["input"], hyperparameters[folder]["target"])

os.makedirs(parserargs["meshoutputfolder"])

targetmesh1 = map_mesh(mappings=mappings, data=data, raise_errors=True, noaffine=True,
                                        remesh=parserargs["remesh"], tmpdir=parserargs["meshoutputfolder"],
                                        update=parserargs["update"])

print("map_mesh called succesfully, now storing meshes")

meshes = {}
meshes["transformed_regaff"] = targetmesh1


if os.path.isdir(parserargs["meshoutputfolder"]) and "test" in parserargs["meshoutputfolder"]:
    os.system("rm -r -v " + parserargs["meshoutputfolder"])






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