#solver for transporting images
from fenics import *
from fenics_adjoint import *
import numpy as np
from dgregister.MRI2FEM import fem2mri

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

def CGTransport(Img, Wind, MaxIter, DeltaT, 
                preconditioner="amg",
                MassConservation=False,
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
        
        solver = KrylovSolver(A, "gmres", preconditioner)
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


