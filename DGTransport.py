#solver for transporting images
from fenics import *
from fenics_adjoint import *

print("Setting parameters parameters['ghost_mode'] = 'shared_facet'")
parameters['ghost_mode'] = 'shared_facet'

# PETScOptions.set("mat_mumps_use_omp_threads", 8)
# PETScOptions.set("mat_mumps_icntl_35", True) # set use of BLR (Block Low-Rank) feature (0:off, 1:optimal)
# PETScOptions.set("mat_mumps_cntl_7", 1e-8) # set BLR relaxation
# PETScOptions.set("mat_mumps_icntl_4", 3)   # verbosity
# PETScOptions.set("mat_mumps_icntl_24", 1)  # detect null pivot rows
# PETScOptions.set("mat_mumps_icntl_22", 0)  # out of core
# #PETScOptions.set("mat_mumps_icntl_14", 250) # max memory increase in %





def Transport(Img, Wind, MaxIter, DeltaT, MassConservation = True, StoreHistory=False, FNameOut="", 
                solver=None, timestepping=None):
    
    # assert timestepping in ["CrankNicolson", "explicitEuler"]

    print("......................................")
    print("Settings in Transport()")
    print("--- solver =", solver)
    print("--- timestepping =", timestepping)
    print("......................................")

    print("parameters['ghost_mode']", parameters['ghost_mode'])

    
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
        raise ValueError("DGTransport: WARNING: CFL = %le", CFL)

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

    a = Constant(1.0/DeltaT)*(inner(v,Img_next)*dx - inner(v, Img_deformed)*dx)
    
    if timestepping == "explicitEuler":
        a = a + Form(Img_deformed)
    elif timestepping == "RungeKutta":
        # in this case we assemble the RHS during the loop
        pass 

    elif timestepping == "CrankNicolson":
        a = a + 0.5*(Form(Img_deformed) + Form(Img_next))
    else:
        raise NotImplementedError

        #a = Constant(1.0/DeltaT)*(inner(v, f_next)*dx - inner(v, Img)*dx) - Form(f_next)
        #a = Constant(1.0/DeltaT)*(inner(v, f_next)*dx - inner(v, Img)*dx) - Form(Img)

    # breakpoint()
    if solver == "krylov":
        A = assemble(lhs(a))
        #solver = LUSolver(A) #needed for Taylor-Test
        solver = KrylovSolver(A, "gmres", "none")
        print("Assembled A, using Krylov solver")
    else:
        
        assert solver == "lu"
        A = assemble(lhs(a))
        solver = LUSolver()
        solver.set_operator(A)
        print("Assembled A, using LU solver")
        # solver = PETScLUSolver(A, "mumps")
        
    
    CurTime = 0.0
    if StoreHistory:
        FOut.write(Img_deformed, CurTime)

    for i in range(MaxIter):
        #solve(a==0, Img_next)

        print("Iteration ", i + 1, "/", MaxIter + 1, "in Transport()")

        if timestepping == "RungeKutta":
            dImg = TrialFunction(Img_deformed.function_space())
            dI = Function(Img_deformed.function_space())
            
            solve(inner(dImg, v)*dx == Form(Img_deformed), dI)
            # A = assemble(lhs(tempA))
            # b = assemble()
            # solve(A, x, b)

            da = Form(Img_deformed  + dI)
            
            system_rhs = rhs(a + da)
        else:
            system_rhs = rhs(a)

        b = assemble(system_rhs)
        b.apply("")
        
        #solver.solve(Img_deformed.vector(), b)
        solver.solve(Img.vector(), b)
        Img_deformed.assign(Img)

        
        CurTime = i*DeltaT
        if StoreHistory:
            FOut.write(Img_deformed, CurTime)

    print("i == MaxIter, Transport() finished")
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
