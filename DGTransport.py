from dolfin import *
parameters['ghost_mode'] = 'shared_facet'
from Pic2Fen import *

def solve(Img, Wind, MaxIter, DeltaT, MassConservation = True, StoreHistory=False, FNameOut=""):
    Space = Img.function_space()
    v = TestFunction(Space)
    if StoreHistory:
        FOut = XDMFFile("output/"+FNameOut+".xdmf")
        FOut.parameters["flush_output"] = True
        FOut.parameters["functions_share_mesh"] = True
        FOut.parameters["rewrite_function_mesh"] = False

    mesh = Space.mesh()

    #Make form:
    n = FacetNormal(mesh)
    def Max0(d):
        return 0.5*(d+abs(d))

    def Flux(f, Wind, n):
        upwind = Max0(inner(Wind,n))
        return f*upwind

    def Form(f):
        #a = inner(v, div(outer(f, -Wind)))*dx
    
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
        print(i)
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
    (mesh, Img, NumData) = Pic2Fenics(FName)
    
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
    StoreHistory = True
    MassConservation = False
    MaxIter = 500
    DeltaT = 5e-4
    x = SpatialCoordinate(mesh)
    Wind = as_vector((0.0, x[1]))
    #Wind = Constant((250.0, 250.0))
    #Wind = project(Wind, VectorFunctionSpace(mesh, "CG", 1))
    
    #Img = project(sqrt(inner(Img, Img)), FunctionSpace(mesh, "DG", 0))
    #Img = project(Img, VectorFunctionSpace(mesh, "CG", 1, NumData))
    Img = project(Img, VectorFunctionSpace(mesh, "DG", 1, NumData))
    Img.rename("img", "")

    Img = solve(Img, MaxIter, DeltaT, MassConservation, StoreHistory, FNameOut)
