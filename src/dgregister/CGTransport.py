#solver for transporting images
from fenics import *
from fenics_adjoint import *
import numpy as np

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass


print_overloaded("Setting parameters parameters['ghost_mode'] = 'shared_facet'")
parameters['ghost_mode'] = 'shared_facet'


q_degree = 6

def CGTransport(Img, Wind, MaxIter, DeltaT, preconditioner="amg", MassConservation=False,
                solver=None, timestepping=None):

    print_overloaded("Calling CGTransport")

    
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
    
    elif timestepping == "RungeKutta":
        # in this case we assemble the RHS during the loop
        dImg = TrialFunction(Img_deformed.function_space())
        dI = Function(Img_deformed.function_space())

        form = inner(dImg, v)*dx 
        Atmp = assemble(form)
        tmpsolver = KrylovSolver(method="cg", preconditioner=preconditioner)
        tmpsolver.set_operators(Atmp, Atmp)
    elif timestepping == "CrankNicolson":
        raise NotImplementedError
        a = a + 0.5*(Form(Img_deformed) + Form(Img_next))
    
    else:
        raise NotImplementedError
    
    A = assemble(lhs(a))

    if solver == "krylov":
        
        solver = KrylovSolver(A, "gmres", preconditioner)
        solver.set_operators(A, A)
        print_overloaded("Assembled A, using Krylov solver")

    b = None
    btmp = None

    for i in range(MaxIter):


        if timestepping == "RungeKutta":
            
            rhstmp = Form(Img_deformed)            
            
            if btmp is None:
                
                btmp = assemble(rhstmp)
            else:
                btmp = assemble(rhstmp, tensor=btmp)

            btmp.apply("")

            tmpsolver.solve(dI.vector(), btmp)

            factor = DeltaT / 2.

            da = Form(Img_deformed + factor * dI)
            
            system_rhs = rhs(a + da)
        else:
            system_rhs = rhs(a)

        if b is None:
            b = assemble(system_rhs)
        else:
            b = assemble(system_rhs, tensor=b)
        solver.solve(Img.vector(), b)

        Img_deformed.assign(Img)

    

    print_overloaded("i == MaxIter, Transport() finished")
    return Img_deformed


