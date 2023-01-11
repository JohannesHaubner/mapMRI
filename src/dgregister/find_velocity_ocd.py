from fenics import *
import dgregister.config as config
# if ocd:
if "optimize" in config.hyperparameters.keys() and (not config.hyperparameters["optimize"]):
    print("Not importing dolfin-adjoint")
else:
    print("Importing dolfin-adjoint")
    from fenics_adjoint import *
import os
import csv
from dgregister.helpers import store_during_callback



set_log_level(LogLevel.CRITICAL)
counter = 0

def csvwrite(name, values, header, mode=None, debug=True): 
   # Default to write, set mode to "a" for append
   if not mode:
       mode = "w"
   # If mode is append, but the file does not exist (yet), write instead
   elif mode == "a" and not os.path.isfile(name):
       mode = "w"
   if debug:
       info(name)
       info(" ".join(str(s) for s in header))
       info(" ".join("%.3e" % g for g in values))
   with open(name, mode) as f:
       writer = csv.writer(f)
       writer.writerow(header)     
       writer.writerow(values)     


def compute_ocd_reduced(c0, c1, tau, alpha, results_dir, hyperparameters, files, starting_guess=None, space="CG", reg="H1", phi_eval=None):
    # c0:
    # c1:
    # tau:   time step
    # D:     diffusion coefficient
    # alpha: regularization parameter
    # space: finite element space for the velocity field ("CG" | "RT" | "BDM")
    # reg:   if "H1", use H1-regularization, else use H(div)
    
    info("Computing OCD via reduced approach")

    # Define mesh and function space for the concentration
    mesh = c0.function_space().mesh()
    C = FunctionSpace(mesh, "CG", 1)
    
    # Space for the convective velocity field phi
    if space == "CG":
        Q = VectorFunctionSpace(mesh, "CG", 1)
    else:
        Q = FunctionSpace(mesh, space, 1)
    
    if phi_eval is not None:
        phi = phi_eval

    elif starting_guess is not None:
        print("-"*80)
        print("Using starting guess")
        phi = starting_guess

    else:
        phi = Function(Q, name="Control")

    # Regularization term
    def R(phi, alpha, mesh):
        if reg == "H1":
            form = 0.5*alpha*(inner(phi, phi) + inner(grad(phi), grad(phi)))*dx(domain=mesh)
        else:
            form = 0.5*alpha*(inner(phi, phi) + inner(div(phi), div(phi)))*dx(domain=mesh)
        return form

    # Define previous solution
    c_ = Function(C)
    c_.assign(c0) # Hack to make dolfin-adjoint happy, maybe just start tape here?

    c2 = Function(C)
    c2.assign(c1)
    
    # Define variational problem
    c = TrialFunction(C)
    d = TestFunction(C)
    F = (1.0/tau*(c - c_)*d + div(c*phi)*d)*dx() # + diffusion(D, c, d)
    a, L = system(F)
    bc = DirichletBC(C, c2, "on_boundary")

    # ... and solve it once
    c = Function(C, name="State")
    solve(a == L, c, bc, solver_parameters={"linear_solver": "mumps"})

    if phi_eval is not None:
        return c, phi_eval

    # Output max values of target and current solution for progress
    # purposes
    info("\max c_1 = %f" % c2.vector().max())
    info("\max c = %f" % c.vector().max())

    # Define the objective functional
    Jd = assemble(0.5*(c - c2)**2*dx(domain=mesh))
    Jreg = assemble(R(phi, alpha, mesh)) # assemble(R(phi, alpha, mesh)*dx(domain=mesh))
    # print(type(jd), type(jreg))
    J = Jd + Jreg
    # J = assemble(j)
    info("J (initial) = %f" % J)
    hyperparameters["Jd_init"] = float(Jd)
    hyperparameters["Jreg_init"] = float(Jreg)
    # Define control field
    m = Control(phi)

    state = Control(c) 

    # # Define call-back for output at each iteration of the optimization algorithm
    # name = lambda s: os.path.join(results_dir, "opts", s)
    # dirname = os.path.join(results_dir, "opts")
    # if not os.path.isdir(dirname):
    #     try:
    #         os.mkdir(dirname)
    #     except FileExistsError:
    #         pass
    # header = ("j", "\max \phi")

    

    # with open(os.path.join(results_dir, "counter.csv"), "w") as f:
    #     info("Updating counter file, counter is now %d " % counter)
    #     writer = csv.writer(f)
    #     writer.writerow((counter,))  

    def eval_cb(j, phi):
        global counter
        counter += 1

        current_pde_solution = state.tape_value()
        current_pde_solution.rename("Img", "")

        store_during_callback(current_iteration=counter, hyperparameters=hyperparameters, files=files, Jd=Jd, Jreg=Jreg, 
                    domainmesh=mesh, velocityField=phi, 
                    current_pde_solution=current_pde_solution, control=None)




        # values = (j, phi.vector().max())
        # # mem = resource.getrusage(resource.RUSAGE_SELF)[2]
        # # info("Current memory usage: %g (MB)" % (mem/1024))
        # info("\tj = %f, \max phi = %f (mm/h)" % values)
        # csvwrite(name("optimization_values.csv"), values, header, mode="a")

        # # Read the optimization counter file, update counter, and
        # # write it back, geez. 
        

        # assert os.path.isdir(results_dir)

        # # with open(os.path.join(results_dir, "counter.csv"), "r") as f:
        # #     reader = csv.reader(f)
        # #     # assert len(reader) > 0

        # #     for row in reader:
        # #         counter = int(row[0])
        
        # # counter += 1

        # # with open(os.path.join(results_dir, "counter.csv"), "w") as f:
        # #     info("Updating counter file, counter is now %d " % counter)
        # #     writer = csv.writer(f)
        # #     writer.writerow((counter,))     

        # # Write current control variable to file in HDF5 and PVD formats
        # file = HDF5File(mesh.mpi_comm(), name("opt_phi_%d.h5" % counter), "w")
        # file.write(phi, "function")
        # file.close()

    # Define reduced functional in terms of J and m
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)

    # Minimize functional
    tol = 1.0e-8
    phi_opt = minimize(Jhat,
                       tol=tol, 
                       options={"gtol": tol, "maxiter": int(hyperparameters["lbfgs_max_iterations"]), "disp": True})
    pause_annotation()

    # Update phi, and do a final solve to compute c
    phi.assign(phi_opt)

    solve(a == L, c, bc, solver_parameters={"linear_solver": "mumps"})
    
    return (c, phi) 

class CFLerror(ValueError):
    '''raise this when CFL is violated'''

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass

print_overloaded("Setting parameters parameters['ghost_mode'] = 'shared_facet'")
parameters['ghost_mode'] = 'shared_facet'

current_iteration = 0

def find_velocity(Img, Img_goal, hyperparameters, files, phi_eval=None, vCG=None, M_lumped_inv=None, starting_guess=None, projection=True):

    if projection:
        VCG = FunctionSpace(Img.function_space().mesh(), "CG", 1)

        Img = project(Img, VCG)
        Img_goal = project(Img_goal, VCG)

    c, phi = compute_ocd_reduced(c0=Img, c1=Img_goal, tau=1, files=files, hyperparameters=hyperparameters, phi_eval=phi_eval, starting_guess=starting_guess,
                                alpha=hyperparameters["alpha"], results_dir=hyperparameters["outputfolder"], space="CG", reg="H1")
    
    # if phi_eval is None:

    #     with XDMFFile(hyperparameters["outputfolder"] + "/Finalstate.xdmf") as xdmf:
    #         xdmf.write_checkpoint(c, "Finalstate", 0.)
        
    #     with XDMFFile(hyperparameters["outputfolder"] + "/Finalvelocity.xdmf") as xdmf:
    #         xdmf.write_checkpoint(phi, "FinalV", 0.)


    return c, phi, None