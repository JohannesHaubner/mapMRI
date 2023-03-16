import argparse
from fenics import *
import os

parser = argparse.ArgumentParser()
parser.add_argument("mesh")
parserargs = vars(parser.parse_args())

assert parserargs["mesh"].endswith(".xml")

m = Mesh(parserargs["mesh"])
bm = BoundaryMesh(m, "exterior")

outfile = parserargs["mesh"].replace(".xml", "_boundary.xml")
File(outfile) << bm

outfile2 = parserargs["mesh"].replace(".xml", "_boundary.xdmf")
with XDMFFile(MPI.comm_world, outfile2) as f:
    f.write(bm)

os.system("meshio-convert " + outfile + " " + outfile.replace(".xml", ".stl"))