#deform a given image and write out a .png in 8 bit

from dolfin import *
from dolfin_adjoint import *
from DGTransport import Transport
from Pic2Fen import Pic2FEM, FEM2Pic
import numpy

import time
import PIL
FName = "shuttle_small.png"
(mesh, Img, NumData) = Pic2FEM(FName)

#Img = project(sqrt(inner(Img, Img)), FunctionSpace(mesh, "DG", 0))
#NumData = 1

#Make Deformation Field
#DG = VectorFunctionSpace(mesh, "DG", 1, NumData)
vCG = VectorFunctionSpace(mesh, "CG", 1)
#n = FacetNormal(mesh)
x = SpatialCoordinate(mesh)
#v = TestFunction(DG)

#mark boundary edges as 1
BoundaryMarker = MeshFunction("size_t", mesh, 1)
for e in edges(mesh):
    if len(e.entities(2)) == 1:
        BoundaryMarker[e.index()] = 1

#make v for defomed data
BC = DirichletBC(vCG, Constant((0.0,0.0)), BoundaryMarker, 1)
Wind_data = Function(vCG)
mylhs = inner(grad(TestFunction(vCG)), grad(TrialFunction(vCG)))*dx
myrhs = inner(as_vector([sin(x[0]/100), cos(x[1]/100)]), TestFunction(vCG))*dx
solve(mylhs == myrhs, Wind_data, BC)
#v created

#set order of DG solver
Order = 1
if NumData == 1:
    Img = project(Img, FunctionSpace(mesh, "DG", Order))
else:
    Img = project(Img, VectorFunctionSpace(mesh, "DG", Order, NumData))
Img.rename("img", "")

DeltaT = 1e-5
MaxIter = 500
Img_deformed = Transport(Img, Wind_data, MaxIter, DeltaT, MassConservation = False)

fout = XDMFFile(MPI.comm_world, "output/Img_Transported.xdmf")
fout.parameters["flush_output"] = True
fout.parameters["rewrite_function_mesh"] = False
fout.write(Img_deformed)

FName = "shuttle_goal.png"
FEM2Pic(Img_deformed, NumData, FName)
