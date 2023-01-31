from dolfin import *

# import dgregister.config as config
# def print_overloaded(*args):
#     if MPI.rank(MPI.comm_world) == 0:
#         # set_log_level(PROGRESS)
#         print(*args)
#     else:
#         pass
# # if ocd:

# if "optimize" in config.hyperparameters.keys() and (not config.hyperparameters["optimize"]):
#     print_overloaded("Not importing dolfin-adjoint")
# else:
#     print_overloaded("Importing dolfin-adjoint")
#     from dolfin_adjoint import *

from dolfin_adjoint import *
from dgregister.config import hyperparameters
assert len(hyperparameters) > 1

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args, flush=True)
    else:
        pass



# def preconditioning(func):

#     smoothen = hyperparameters["smoothen"]

#     print("smoothen=", smoothen)


#     if not smoothen:
#         c = func.copy()
#         C = c.function_space()
#         dim = c.geometric_dimension()
#         BC=DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
#         BC.apply(c.vector())
#     else:
#         C = func.function_space()
#         dim = func.geometric_dimension()
#         BC = DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
#         c = TrialFunction(C)
#         psi = TestFunction(C)
#         a = inner(grad(c), grad(psi)) * dx
#         L = inner(func, psi) * dx
#         c = Function(C)
#         solve(a == L, c, BC)
#     return c





class Preconditioning():

    def __init__(self) -> None:
        # self.smoothen = hyperparameters["smoothen"]
        # self.hyperparameters = hyperparameters

        # print("Class is initialized, MPI rank=" + str(MPI.rank(MPI.comm_world)), flush=True)

        pass

    def __call__(self, func):

        if not hyperparameters["smoothen"]:

            # print_overloaded("applying BC to func in Preconditioning()")
            cc = func.copy()

            # breakpoint()

            # print_overloaded("Debugging: cc ", cc.vector()[:].min(), cc.vector()[:].max(), cc.vector()[:].mean())
            C = cc.function_space()
            dim = cc.geometric_dimension()
            BC=DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
            BC.apply(cc.vector())

            # print_overloaded("Debugging: cc ", cc.vector()[:].min(), cc.vector()[:].max(), cc.vector()[:].mean())

            return cc

        else:
            C = func.function_space()
            dim = func.geometric_dimension()

            if not hasattr(self, "BC"):

                self.BC = DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
                self.c = TrialFunction(C)
                self.psi = TestFunction(C)
                self.cc = Function(C)
                # self.matrix

            if not hasattr(self, "solver"):
                a = inner(grad(self.c), grad(self.psi)) * dx
                self.A = assemble(a)

                print_overloaded("Assembled A in Preconditioning() with ", func)

                self.BC.apply(self.A)
                print("Applying BC (at init, i.e. only once?) to func=", func, flush=True)
            
            L = inner(func, self.psi) * dx
            
            
            tmp = assemble(L)
            self.BC.apply(tmp)
            
            # self.BC.apply(self.A)
            # print_overloaded("Applying BC! to func=", func)

            # solve(a == L, c, BC)

            if not hasattr(self, "solver"):

                if hyperparameters["solver"] == "lu":
                    
                    self.solver = LUSolver()
                    self.solver.set_operator(self.A)

                    print_overloaded("Created LU solver in Preconditioning()")

                elif hyperparameters["solver"] == "krylov":
                    # self.solver = PETScKrylovSolver(method="gmres", preconditioner=self.hyperparameters["preconditioner"])
                    self.solver = PETScKrylovSolver("gmres", hyperparameters["preconditioner"])
                    self.solver.set_operators(self.A, self.A)

                    print_overloaded("Created Krylov solver (For the first time ? ) in Preconditioning() with ", func)


            # BC.apply(self.A)
            # x = args[0]
            # b = args[1]

            print_overloaded("Solving in preconditioning, func=", func)
            self.solver.solve(self.cc.vector(), tmp)

            return self.cc



preconditioning = Preconditioning()


# def preconditioning(func):

#     if not hyperparameters["smoothen"]:

#         # print_overloaded("applying BC to func in Preconditioning()")
#         cc = func.copy()

#         # breakpoint()

#         # print_overloaded("Debugging: cc ", cc.vector()[:].min(), cc.vector()[:].max(), cc.vector()[:].mean())
#         C = cc.function_space()
#         dim = cc.geometric_dimension()
#         BC=DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
#         BC.apply(cc.vector())
#     else:
#         C = func.function_space()
#         dim = func.geometric_dimension()



#         BC = DirichletBC(C, Constant((0.0,)*dim), "on_boundary")
#         c = TrialFunction(C)
#         psi = TestFunction(C)
#         cc = Function(C)
#         # self.matrix

#         a = inner(grad(c), grad(psi)) * dx
#         # a = inner(grad(c), grad(psi)) * dx
#         A = assemble(a)
#         print_overloaded("Assembled A in Preconditioning() with", func)
        
#         L = inner(func, psi) * dx
        
        
#         tmp = assemble(L)
#         BC.apply(tmp)
        
#         BC.apply(A)
#         # solve(a == L, c, BC)


#         solver = PETScKrylovSolver("gmres", hyperparameters["preconditioner"])
#         solver.set_operators(A, A)

#         print_overloaded("Created Krylov solver in Preconditioning() with ", func)


#         # BC.apply(self.A)
#         # x = args[0]
#         # b = args[1]

        
#         solver.solve(cc.vector(), tmp)

#     return cc


# preconditioning = lambda x: x