
from __future__ import print_function
from fenics import *
from fenics_adjoint import *
import argparse
import resource

set_log_level(LogLevel.CRITICAL)


mesh = UnitSquareMesh(64, 64)
V = FunctionSpace(mesh, "CG", 1)

dt = 0.001


def main(ic, annotate=False):
    u_prev = ic.copy(deepcopy=True)
    # u_next = ic.copy(deepcopy=True)
    
    u_next = TrialFunction(V)
    u = Function(V)
    
    u_mid = Constant(0.5) * u_prev + Constant(0.5) * u_next


    T = 0.1
    t = 0.0

    v = TestFunction(V)

    states = [ic.copy(deepcopy=True)]
    times = [float(t)]

    timestep = 0

    F = inner((u_next - u_prev) / Constant(dt), v) * dx + inner(grad(u_mid), grad(v)) * dx
    # solve(F == 0, u_next, J=derivative(F, u_next), annotate=annotate)

    f = assemble(lhs(F))
    b = assemble(rhs(F))

    solver = PETScKrylovSolver("gmres", "amg")
    solver.set_operators(f, f)

    while t < T:
        print("Solving for t == %s" % (t + dt))
        
        # F = inner((u_next - u_prev) / Constant(dt), v) * dx + inner(grad(u_mid), grad(v)) * dx
        # solve(F == 0, u_next, J=derivative(F, u_next), annotate=annotate)
        
        # # x = args[0]
        # b = args[1]
        # print(type(u_next), type(b))
        solver.solve(u.vector(), b) # , J=derivative(F, u_next), annotate=annotate)
        
        u_prev.assign(u, annotate=annotate)

        t += dt
        timestep += 1
        # states.append(u_next.copy(deepcopy=True, annotate=False))
        # times.append(float(t))

    return (times, states, u_prev)


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("--lbfgs_max_iterations", type=float, default=50)
    parser.add_argument("--maxcor", default=10, type=int)

    hyperparameters = vars(parser.parse_args())


    true_ic = interpolate(Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1), V)
    (times, true_states, u) = main(true_ic, annotate=False)

    guess_ic = interpolate(Expression("15 * x[0] * (1 - x[0]) * x[1] * (1 - x[1])", degree=1), V)
    (times, computed_states, u) = main(guess_ic, annotate=True)

    combined = zip(times, true_states, computed_states)

    alpha = Constant(1.0e-7)
    J = assemble(
        sum(inner(true - computed, true - computed) * dx for (time, true, computed) in combined if time >= 0.01)
        + alpha * inner(grad(guess_ic), grad(guess_ic)) * dx)

    m = Control(guess_ic)

    m_ex = Function(V, name="Temperature")
    # viz = File("output/iterations.pvd")

    def derivative_cb(j, dj, m):
        m_ex.assign(m)
        # viz << m_ex

    rf = ReducedFunctional(J, m)
    current_iteration = 1
    fac = 1


    def cb(*args, **kwargs):
        global current_iteration
        global fac
        
        mem = resource.getrusage(resource.RUSAGE_SELF)[2] / (1e6*1024)

        if current_iteration == 1:
            fac = mem
        

        current_iteration += 1
        print("Memory (TB)", mem, "current_iteration", current_iteration, "process", str(MPI.rank(MPI.comm_world)))

        return 

    minimize(rf,  method = 'L-BFGS-B', options = {"iprint": 0, "disp": None, "maxiter": hyperparameters["lbfgs_max_iterations"],
            # "maxls": 1,  "ftol": 0, "gtol": 0, 
            "maxcor": hyperparameters["maxcor"]}, tol=1e-16, callback = cb)


    # problem = MinimizationProblem(rf)
    # parameters = {'maximum_iterations': 50}

    # solver = IPOPTSolver(problem, parameters=parameters)
    # rho_opt = solver.solve()