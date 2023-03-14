from dolfin import *
from dolfin_adjoint import *

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        print(*args, flush=True)
    else:
        pass


class Preconditioning():

    def __init__(self) -> None:
        self.tmp = None
        self.A = None
        

    def __call__(self, func):

        C = func.function_space()
        dim = func.geometric_dimension()
        BC = DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
        c = TrialFunction(C)
        psi = TestFunction(C)
        
        omega = 0
        epsilon = 1

        omega = 0.5
        epsilon = 0.5
        print_overloaded("Using non-default omega=", omega, "epsilon=", epsilon, "in preconditioning_overloaded")


        if not hasattr(self, "solver"):

            a = omega * inner(c, psi) * dx + epsilon * inner(grad(c), grad(psi)) * dx

            if self.A is None:

                self.A = assemble(a)

            else:
                self.A = assemble(a, tensor=self.A)

            BC.apply(self.A)

            self.solver = PETScKrylovSolver("gmres", "amg")
            self.solver.set_operators(self.A, self.A)
            print_overloaded("Created Krylov solver in Preconditioning()")

        L = inner(func, psi) * dx

        if self.tmp is None:
            self.tmp = assemble(L)
        else:
            self.tmp = assemble(L, tensor=self.tmp)
        
        BC.apply(self.tmp)

        cc = Function(C)
        self.solver.solve(cc.vector(), self.tmp)

        return cc


preconditioning = Preconditioning()