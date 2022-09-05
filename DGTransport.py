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
    
    def Max0smoothed(d, k=1):
        return d/(1+exp(-Constant(k)*d))

    #Scheme = "Central" #needed for Taylor-Test
    #Scheme = "Upwind"
    Scheme = "smoothedUpwind"
    
    def Flux(f, Wind, n):
        if Scheme == "Central":
            flux = 0.5*inner(Wind, n)
        if Scheme == "Upwind":
            flux = Max0(inner(Wind,n))
        if Scheme == "smoothedUpwind":
            flux = Max0smoothed(inner(Wind, n))
        return f*flux
        
    def FluxB(f, Wind, n):
        if Scheme == "Central":
            return f*inner(Wind,n)
        if Scheme == "Upwind":
            return f*Max0(inner(Wind,n))
        if Scheme == "smoothedUpwind":
            return f*Max0smoothed(inner(Wind, n))


    def Form(f):
        #a = inner(v, div(outer(f, Wind)))*dx
    
        a = -inner(grad(v), outer(f, Wind))*dx
        a += inner(jump(v), jump(Flux(f, Wind, n)))*dS
        a += inner(v, FluxB(f, Wind, n))*ds
    
        if MassConservation == False:
            a -= inner(v, div(Wind)*f)*dx
        return a

    Img_next = TrialFunction(Img.function_space())
    #Img_next = Function(Img.function_space())
    #Img_next.rename("img", "")
    Img_deformed = Function(Img.function_space())
    Img_deformed.assign(Img)
    Img_deformed.rename("Img", "")

    a = Constant(1.0/DeltaT)*(inner(v,Img_next)*dx - inner(v, Img_deformed)*dx) + 0.5*(Form(Img_deformed) + Form(Img_next))

    #a = Constant(1.0/DeltaT)*(inner(v, f_next)*dx - inner(v, Img)*dx) - Form(f_next)
    #a = Constant(1.0/DeltaT)*(inner(v, f_next)*dx - inner(v, Img)*dx) - Form(Img)

    A = assemble(lhs(a))
    #solver = LUSolver(A) #needed for Taylor-Test
    solver = KrylovSolver(A, "gmres", "none")
    
    CurTime = 0.0
    if StoreHistory:
        FOut.write(Img_deformed, CurTime)

    for i in range(MaxIter):
        #solve(a==0, Img_next)

        b = assemble(rhs(a))
        b.apply("")
        
        #solver.solve(Img_deformed.vector(), b)
        solver.solve(Img.vector(), b)
        Img_deformed.assign(Img)
        
        CurTime = i*DeltaT
        if StoreHistory:
            FOut.write(Img_deformed, CurTime)
    return Img_deformed

if __name__ == "__main__":
    #create on the fly
    FName = "shuttle_small.png"
    from Pic2Fen import Pic2FEM, FEM2Pic
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
    
    FNameOut = "img_DG"
    FNameOut = "output/"+FNameOut+".xdmf"
    StoreHistory = True
    MassConservation = False
    MaxIter = 300
    DeltaT = 2e-4
    
    x = SpatialCoordinate(mesh)
    Wind = as_vector((0.0, x[1]))

    #Img = project(sqrt(inner(Img, Img)), FunctionSpace(mesh, "DG", 0))
    #Img = project(Img, VectorFunctionSpace(mesh, "CG", 1, NumData))
    Img = project(Img, VectorFunctionSpace(mesh, "DG", 1, NumData))
    Img.rename("img", "")

    Img_deformed = Transport(Img, Wind, MaxIter, DeltaT, MassConservation, StoreHistory, FNameOut)
    File("output/DGTransportFinal.pvd") << Img_deformed
    FEM2Pic(Img_deformed, NumData, "output/DGTransportFinal.png")
