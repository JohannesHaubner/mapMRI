#solver for transporting images

from dolfin import *
#from dolfin_adjoint import *
#parameters['ghost_mode'] = 'shared_facet'

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

    def Form(u):
        from numpy import zeros as npzeros
        #weak form
        re = Constant(0.0) #physical diffusion
        f = Constant(npzeros(NumData)) #source term
        a = inner(v, dot(grad(u), Wind))*dx + re*inner(grad(v), grad(u))*dx
        
        # Add SUPG stabilisation terms
        # Residual
        r = dot(grad(u), Wind) - re*div(grad(u))# - f
        vnorm = sqrt(dot(Wind, Wind))
        a += Constant(0.01)*(h/(2.0*vnorm))*inner(dot(grad(v), Wind), r)*dx
        return a

    Img_next = TrialFunction(Img.function_space())
    #Img_next = Function(Img.function_space())
    #Img_next.rename("img", "")
    Img_deformed = Function(Img.function_space())
    Img_deformed.assign(Img)
    Img_deformed.rename("Img", "")

    a = Constant(1.0/DeltaT)*(inner(v,Img_next) - inner(v,Img_deformed))*dx + Form(0.5*(Img_deformed + Img_next))
    A = assemble(lhs(a))
    solver = LUSolver(A) #needed for Taylor-Test
    #solver = KrylovSolver(A, "gmres", "none")
    
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
    
    Img = project(Img, VectorFunctionSpace(mesh, "CG", 2, NumData))
    #Make BW
    #Img = project(sqrt(inner(Img, Img)), FunctionSpace(mesh, "CG", 1))
    #Img.rename("img_SUPG", "")
    #NumData = 1
    
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
    
    FNameOut = "img_SUPG"
    FNameOut = "output/"+FNameOut+".xdmf"
    StoreHistory = True
    MassConservation = False
    MaxIter = 300
    DeltaT = 2e-4
    
    x = SpatialCoordinate(mesh)
    Wind = as_vector((0.0, x[1]))

    Img_deformed = Transport(Img, Wind, MaxIter, DeltaT, MassConservation, StoreHistory, FNameOut)
    File("output/SUPGTransportFinal.pvd") << Img_deformed
    FEM2Pic(Img_deformed, NumData, "output/SUPGTransportFinal.png")
