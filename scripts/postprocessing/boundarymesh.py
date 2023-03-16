import argparse
from fenics import *
import os
import SVMTK as svmtk
import meshio

parser = argparse.ArgumentParser()
parser.add_argument("mesh")
parser.add_argument("--fix", action="store_true", default=False)
parser.add_argument("--noredo", action="store_true", default=False)
parserargs = vars(parser.parse_args())

assert parserargs["mesh"].endswith(".xml")
outfile = parserargs["mesh"].replace(".xml", "_boundary.xml")

if (not parserargs["noredo"]):

    m = Mesh(parserargs["mesh"])
    bm = BoundaryMesh(m, "exterior")

    
    File(outfile) << bm

    outfile2 = parserargs["mesh"].replace(".xml", "_boundary.xdmf")
    with XDMFFile(MPI.comm_world, outfile2) as f:
        f.write(bm)

    os.system("meshio-convert " + outfile + " " + outfile.replace(".xml", ".stl"))

if parserargs["fix"]:

    stlfile = outfile.replace(".xml", ".stl")

    fixedstlfile = stlfile.replace(".stl", "_fixed.stl")

    if (not parserargs["noredo"]):

        surface = svmtk.Surface(stlfile)

        # Remesh surface
        surface.isotropic_remeshing(1, 3, False)

        surface.smooth_taubin(5)

        surface.fill_holes()

        # Separate narrow gaps
        # Default argument is -0.33. 
        surface.separate_narrow_gaps(-0.33)
        

        surface.save(fixedstlfile)

    
    mm = meshio.read(fixedstlfile)
    mm.write(fixedstlfile.replace(".stl", ".xml"))
    mm.write(fixedstlfile.replace(".stl", ".xdmf"))


    # print("Done with surface fixes")

    # os.system("meshio-convert " + fixedstlfile + " " + fixedstlfile.replace(".stl", ".xml"))
    # os.system("meshio-convert " + fixedstlfile + " " + fixedstlfile.replace(".stl", ".xdmf"))