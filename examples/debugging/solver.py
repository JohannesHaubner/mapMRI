from fenics import *
import resource
import gc

n = 32

m = UnitSquareMesh(n, n)

V = FunctionSpace(m, "CG", 1)

u = Function(V)
# u2 = TestFunction(V)
# u1 = TrialFunction(V)

# f = interpolate(Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3), V)


# # Define boundary condition
# u_D = Expression('0', degree=2)

# def boundary(x, on_boundary):
#     return on_boundary

# bc = DirichletBC(V, u_D, boundary)

# a = inner(grad(u1), grad(u2)) * dx - f * u2 * dx

# A = assemble(lhs(a))
# b = assemble(rhs(a))

# bc.apply(A)
# bc.apply(b)

# # print(krylov_solver_methods())
# solver = KrylovSolver(method="cg", preconditioner="amg")
# solver.set_operators(A, A)

# solver.solve(u.vector(), b)#, "cg", "hypre_amg")

# # print(u.vector()[:])

mem = resource.getrusage(resource.RUSAGE_SELF)[2]
print("Memory (TB)", (mem/(1e6*1024)), "current_iteration", "-7", "process", str(MPI.rank(MPI.comm_world)))



class Projector():
    def __init__(self, V):
        self.v = TestFunction(V)
        u = TrialFunction(V)
        form = inner(u, self.v)*dx
        self.A = assemble(form)# , annotate=False)
        # self.solver = LUSolver(self.A)
        self.V = V
        self.solver = KrylovSolver()
        self.solver.set_operators(self.A, self.A)
        # self.uh = Function(V)
    
    
    def project(self, f):
        L = inner(f, self.v)*dx
        b = assemble(L) # , annotate=False)
        
        uh = Function(self.V)
        self.solver.solve(uh.vector(), b)
        
        return uh

projector = Projector(FunctionSpace(m, "CG", 4))

u_data = projector.project(u)

mem = resource.getrusage(resource.RUSAGE_SELF)[2]
print("Memory (TB)", format(mem/(1e6*1024), ".1e"), "current_iteration", "-7", "process", str(MPI.rank(MPI.comm_world)))

del projector, u_data, u

gc.collect()

mem = resource.getrusage(resource.RUSAGE_SELF)[2]
print("Memory (TB)", format(mem/(1e6*1024), ".1e"), "current_iteration", "-7", "process", str(MPI.rank(MPI.comm_world)))



print("done")