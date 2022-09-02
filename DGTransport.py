#solver for transporting images

from dolfin import *
from dolfin_adjoint import *
parameters['ghost_mode'] = 'shared_facet'

def Transport(Img, Wind, MaxIter, DeltaT, MassConservation = True, StoreHistory=False, FNameOut=""):
    Space = Img.function_space()
    v = TestFunction(Space)
    if StoreHistory:
        FOut = XDMFFile(FNameOut)
        FOut.parameters["flush_output"] = True
        FOut.parameters["functions_share_mesh"] = True
        FOut.parameters["rewrite_function_mesh"] = False

    mesh = Space.mesh()
    #compute CFL number
    h = CellDiameter(mesh)
    CFL = project(sqrt(inner(Wind, Wind))*Constant(DeltaT)/h, FunctionSpace(mesh, "DG", 0))
    if(CFL.vector().max() > 1.0):
        print("DGTransport: WARNING: CFL = %le", CFL)

    #Make form:
    n = FacetNormal(mesh)
    def Max0(d):
        return 0.5*(d+abs(d))

    def Flux(f, Wind, n):
        upwind = Max0(inner(Wind,n))
        return f*upwind

    def Form(f):
        #a = inner(v, div(outer(f, Wind)))*dx
    
        a = -inner(grad(v), outer(f, Wind))*dx
        a += inner(jump(v), jump(Flux(f, Wind, n)))*dS
        a += inner(v, Flux(f, Wind, n))*ds
    
        if MassConservation == False:
            a -= inner(v, div(Wind)*f)*dx
        return a

    Img_next = TrialFunction(Img.function_space())
    #Img_next = Function(Img.function_space())
    #Img_next.rename("img", "")

    a = Constant(1.0/DeltaT)*(inner(v,Img_next)*dx - inner(v, Img)*dx) + 0.5*(Form(Img) + Form(Img_next))

    #a = Constant(1.0/DeltaT)*(inner(v, f_next)*dx - inner(v, Img)*dx) - Form(f_next)
    #a = Constant(1.0/DeltaT)*(inner(v, f_next)*dx - inner(v, Img)*dx) - Form(Img)

    A = assemble(lhs(a))
    #solver = LUSolver()
    solver = KrylovSolver("gmres", "none")
    solver.set_operator(A)

    CurTime = 0.0
    if StoreHistory:
        FOut.write(Img, CurTime)

    for i in range(MaxIter):
        #solve(a==0, Img_next)

        b = assemble(rhs(a))
        b.apply("")
        solver.solve(Img.vector(), b)
        CurTime = i*DeltaT
        if StoreHistory:
            FOut.write(Img, CurTime)
    return Img

if __name__ == "__main__":
    #create on the fly
    FName = "shuttle_small.png"
    from Pic2Fen import Pic2FEM
    (mesh, Img, NumData) = Pic2FEM(FName)

    """
    #read from file
    FIn = HDF5File(MPI.comm_world, FName+".h5", 'r')
    mesh = Mesh()
    FIn.read(mesh, "mesh", False)
    Space = VectorFunctionSpace(mesh, "DG", 0)
    Img = Function(Space)
    FIn.read(Img, "Data1")
    """

    """
    #make artificial
    mesh = UnitSquareMesh(100,100)
    x = SpatialCoordinate(mesh)
    Img = project(x[0], FunctionSpace(mesh, "DG", 0))
    NumData = 1
    """
    
    FNameOut = "img"
    FNameOut = "output/"+FNameOut+".xdmf"
    StoreHistory = True
    MassConservation = False
    MaxIter = 500
    DeltaT = 1e-4
    
    x = SpatialCoordinate(mesh)
    Wind = as_vector((0.0, x[1]))

    #Img = project(sqrt(inner(Img, Img)), FunctionSpace(mesh, "DG", 0))
    #Img = project(Img, VectorFunctionSpace(mesh, "CG", 1, NumData))
    Img = project(Img, VectorFunctionSpace(mesh, "DG", 1, NumData))
    Img.rename("img", "")

    Img = Transport(Img, Wind, MaxIter, DeltaT, MassConservation = True, StoreHistory=True, FNameOut=FNameOut)
