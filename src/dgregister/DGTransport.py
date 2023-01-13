#solver for transporting images
from fenics import *

import dgregister.config as config
# if ocd:
if "optimize" in config.hyperparameters.keys() and (not config.hyperparameters["optimize"]):
    print("Not importing dolfin-adjoint")
else:
    print("Importing dolfin-adjoint")

    from fenics_adjoint import *


def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass


print_overloaded("Setting parameters parameters['ghost_mode'] = 'shared_facet'")
parameters['ghost_mode'] = 'shared_facet'






def CGTransport(Img, Wind, MaxIter, DeltaT, hyperparameters, MassConservation=False, StoreHistory=False, FNameOut="",
                solver=None, timestepping=None):

    print_overloaded("Calling CGTransport")

    if not timestepping == "explicitEuler":
        raise NotImplementedError
    
    Space = Img.function_space()
    v = TestFunction(Space)
    Img_next = TrialFunction(Img.function_space())
    Img_deformed = Function(Img.function_space())
    Img_deformed.assign(Img)

    def Form(f):
        #a = inner(v, div(outer(f, Wind)))*dx
    
        a = -inner(grad(v), outer(f, Wind))*dx
        # a += inner(jump(v), jump(Flux(f, Wind, n)))*dS
        # a += inner(v, FluxB(f, Wind, n))*ds
    
        if MassConservation == False:
            a -= inner(v, div(Wind)*f)*dx
        return a

    a = Constant(1.0/DeltaT)*(inner(v,Img_next)*dx - inner(v, Img_deformed)*dx)
    
    if timestepping == "explicitEuler":
        a = a + Form(Img_deformed)
    else:
        raise NotImplementedError

    A = assemble(lhs(a))

    if solver == "krylov":
        
        solver = KrylovSolver(A, "gmres", hyperparameters["preconditioner"])
        solver.set_operators(A, A)
        print_overloaded("Assembled A, using Krylov solver")

    for i in range(MaxIter):
        #solve(a==0, Img_next)

        print_overloaded("Iteration ", i + 1, "/", MaxIter, "in Transport()")

        system_rhs = rhs(a)

        b = assemble(system_rhs)
        b.apply("")
        
        #solver.solve(Img_deformed.vector(), b)
        solver.solve(Img.vector(), b)
        Img_deformed.assign(Img)

        assert norm(Img_deformed) > 0

    

    print_overloaded("i == MaxIter, Transport() finished")
    return Img_deformed



def DGTransport(Img, Wind, MaxIter, DeltaT, hyperparameters, MassConservation=False, StoreHistory=False, FNameOut="",
                solver=None, timestepping=None):
    
    # assert timestepping in ["CrankNicolson", "explicitEuler"]

    print_overloaded("......................................")
    print_overloaded("Settings in Transport()")
    print_overloaded("--- solver =", solver)
    print_overloaded("--- timestepping =", timestepping)
    print_overloaded("......................................")

    print_overloaded("parameters['ghost_mode']", parameters['ghost_mode'])

    
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
    elif timestepping == "RungeKuttaBug":
        # in this case we assemble the RHS during the loop
        pass 
    elif timestepping == "CrankNicolson":
        a = a + 0.5*(Form(Img_deformed) + Form(Img_next))
    else:
        raise NotImplementedError

        #a = Constant(1.0/DeltaT)*(inner(v, f_next)*dx - inner(v, Img)*dx) - Form(f_next)
        #a = Constant(1.0/DeltaT)*(inner(v, f_next)*dx - inner(v, Img)*dx) - Form(Img)

    A = assemble(lhs(a))

    if solver == "krylov":
        
        solver = KrylovSolver(A, "gmres", hyperparameters["preconditioner"])
        solver.set_operators(A, A)
        print_overloaded("Assembled A, using Krylov solver")
    
    elif solver == "lu":
        solver = LUSolver()
        solver.set_operator(A)
        print_overloaded("Assembled A, using LU solver")

        # solver = PETScLUSolver(A, "mumps")
    else:
        raise NotImplementedError()
    
    CurTime = 0.0
    if StoreHistory:
        FOut.write(Img_deformed, CurTime)

    for i in range(MaxIter):
        #solve(a==0, Img_next)

        print_overloaded("Iteration ", i + 1, "/", MaxIter, "in Transport()")

        if timestepping == "RungeKutta" or timestepping == "RungeKuttaBug":
            dImg = TrialFunction(Img_deformed.function_space())
            dI = Function(Img_deformed.function_space())
            
            solve(inner(dImg, v)*dx == Form(Img_deformed), dI)
            # A = assemble(lhs(tempA))
            # b = assemble()
            # solve(A, x, b)

            # BZ: potentially factor dt / 2 missing in front of dI ?
            factor = DeltaT / 2.

            if timestepping == "RungeKuttaBug":
                factor = 1

            da = Form(Img_deformed + factor * dI)
            
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

    print_overloaded("i == MaxIter, Transport() finished")
    return Img_deformed

if __name__ == "__main__":
    #create on the fly
    FName = "shuttle_small.png"
    from dgregister.Pic2Fen import Pic2FEM, FEM2Pic
    (mesh, Img, NumData) = Pic2FEM(FName)
    
    FNameOut = "img_DG"
    FNameOut = "output/"+FNameOut+".xdmf"
    StoreHistory = True
    MassConservation = False
    MaxIter = 300
    DeltaT = 2e-4
    
    x = SpatialCoordinate(mesh)
    Wind = as_vector((0.0, x[1]))

    Img = project(Img, VectorFunctionSpace(mesh, "DG", 1, NumData))
    Img.rename("img", "")

    Img_deformed = DGTransport(Img, Wind, MaxIter, DeltaT, MassConservation, StoreHistory, FNameOut)
    File("output/DGTransportFinal.pvd") << Img_deformed
    FEM2Pic(Img_deformed, NumData, "output/DGTransportFinal.png")
