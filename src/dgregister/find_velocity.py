from fenics import *
from fenics_adjoint import *
from dgregister.DGTransport import DGTransport
from dgregister.transformation_overloaded import transformation
from dgregister.preconditioning_overloaded import preconditioning
from dgregister.tukey import tukey
import time, json
import numpy as np
import resource
# import nibabel
# from dgregister.MRI2FEM import fem2mri

from dgregister.helpers import store_during_callback


set_log_level(LogLevel.CRITICAL)

# class Projector():
#     def __init__(self, V):
#         self.v = TestFunction(V)
#         u = TrialFunction(V)
#         form = inner(u, self.v)*dx
#         self.A = assemble(form, annotate=False)
#         self.solver = LUSolver(self.A)
#         self.uh = Function(V)
#     def project(self, f):
#         L = inner(f, self.v)*dx
#         b = assemble(L, annotate=False)
#         self.solver.solve(self.uh.vector(), b)
#         return self.uh




class CFLerror(ValueError):
    '''raise this when CFL is violated'''

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args, flush=True)
    else:
        pass

print_overloaded("Setting parameters parameters['ghost_mode'] = 'shared_facet'")
parameters['ghost_mode'] = 'shared_facet'

current_iteration = 0



def find_velocity(starting_image, Img_goal, vCG, M_lumped_inv, hyperparameters, files): #, starting_guess):

    vol = assemble(1*dx(starting_image.function_space().mesh()))

    hyperparameters["vol"] = vol

    set_working_tape(Tape())

    # initialize control
    l2_controlfun = Function(vCG)

    # if (hyperparameters["starting_guess"] is not None):
    #     # raise NotImplementedError("Double check that you are reading the correct file")
    #     l2_controlfun.assign(starting_guess)
    #     print_overloaded("*"*20, "assigned starting guess to control")
    #     assert norm(l2_controlfun) > 0


    if hyperparameters["preconditioning"] == "none":

        # def preconditioning(x):
        #     print_overloaded("Not doing preconditioning")
        #     return x 
    
        # def transformation(x, y):
        #     print_overloaded("Not doing transform")
        #     return x 

        control_L2 = l2_controlfun
        velocity = control_L2
        print_overloaded("Setting velocity = l2_controlfun")
    else:
        #######################
        #######################
        assert hyperparameters["preconditioning"] == "preconditioning"
        #######################
        #######################
        if hyperparameters["smoothen"]:
            print_overloaded("Transforming l2 control to L2 control")
            control_L2 = transformation(l2_controlfun, M_lumped_inv)
        else:
            control_L2 = l2_controlfun

        print_overloaded("Preconditioning L2_controlfun, name=", control_L2)
        velocity = preconditioning(control_L2)

    l2_controlfun.rename("control_l2", "")
    control_L2.rename("control_L2", "")
    velocity.rename("velocity", "")

    print_overloaded("Running Transport() with dt = ", hyperparameters["DeltaT"])

    mem = resource.getrusage(resource.RUSAGE_SELF)[2]
    print("Memory (TB)", (mem/(1e6*1024)), "current_iteration", "-6", "process", str(MPI.rank(MPI.comm_world)))

    Img_deformed = DGTransport(starting_image, velocity, preconditioner=hyperparameters["preconditioner"],
                            MaxIter=hyperparameters["max_timesteps"], DeltaT=hyperparameters["DeltaT"], timestepping=hyperparameters["timestepping"], 
                            solver=hyperparameters["solver"], MassConservation=hyperparameters["MassConservation"], reassign=hyperparameters["reassign"])

    mem = resource.getrusage(resource.RUSAGE_SELF)[2]
    print("Memory (TB)", (mem/(1e6*1024)), "current_iteration", "-5", "process", str(MPI.rank(MPI.comm_world)))

    # if hyperparameters["projector"]:
    #     Img_deformed = projectorU.project(dot(velocity, ny))
    # else:
    #     Img_deformed = project(dot(velocity, ny), Img.function_space())


    # solve forward and evaluate objective
    alpha = Constant(hyperparameters["alpha"]) #regularization
    state = Control(Img_deformed)  # The Control type enables easy access to tape values after replays.
    cont = Control(l2_controlfun)


    if hyperparameters["tukey"]:

        def tukeyloss(x, y, hyperparameters):

            tukey_c = hyperparameters["tukey_c"]

            residual = x - y

            # mean_residual = assemble(residual * dx) / vol

            # std_residual = sqrt( assemble((residual - mean_residual) ** 2 * dx) / vol )


            with stop_annotating():
                arrname = hyperparameters["outputfolder"] + "/normalized_residual" + str(MPI.rank(MPI.comm_world)) + ".npy"
                resvec = x.vector()[:] - y.vector()[:] #  - mean_residual) / std_residual
                np.save(arrname, resvec)

                print_overloaded("Stored residuals")
            
            loss = tukey(x=residual, c=tukey_c)
            
            return loss

        print_overloaded("Using tukey loss")
        loss = tukeyloss(x=Img_deformed, y=Img_goal, hyperparameters=hyperparameters)
        Jd = assemble(0.5 * loss * dx)


        mem = resource.getrusage(resource.RUSAGE_SELF)[2]
        print_overloaded("Current memory usage (after assembling tukey loss): %g (MB)" % (mem/1024))

        with stop_annotating():
            l2loss = assemble(0.5 * (Img_deformed - Img_goal) ** 2 * dx)

        mem = resource.getrusage(resource.RUSAGE_SELF)[2]
        print_overloaded("Current memory usage (after assembling l2 loss): %g (MB)" % (mem/1024))

    else:
        Jd = assemble(0.5 * (Img_deformed - Img_goal) ** 2 * dx)
        
        with stop_annotating():
            l2loss = Jd

    assert Jd > 0
    

    print_overloaded("Assembled error between transported image and target, Jdata=", Jd)
    print_overloaded("L2 error between transported image and target, Jdata_L2=", l2loss)

    Jreg = assemble(alpha*(control_L2)**2*dx) # (domain=Img.function_space().mesh()))
    
    print_overloaded("At init:")
    print_overloaded("Jd", Jd)
    print_overloaded("Reg", Jreg)

    J = Jd + Jreg


    if MPI.rank(MPI.comm_world) == 0:
    
        with open(files["lossfile"], "a") as myfile:
            myfile.write(str(float(Jd))+ ", ")
        with open(files["l2lossfile"], "a") as myfile:
            myfile.write(str(float(l2loss))+ ", ")

    


    
    hyperparameters["Jd_init"] = float(Jd)
    hyperparameters["Jreg_init"] = float(Jreg)
    hyperparameters["JL2_init"] = float(l2loss)

    print_overloaded("Assembled functional, J=", J)

    Jhat = ReducedFunctional(J, cont)

    mem = resource.getrusage(resource.RUSAGE_SELF)[2]
    print("Memory (TB)", (mem/(1e6*1024)), "current_iteration", "-4", "process", str(MPI.rank(MPI.comm_world)))

    mem = resource.getrusage(resource.RUSAGE_SELF)[2]
    print_overloaded("Current memory usage (after creating reduced functional): %g (MB)" % (mem/1024))

    if hyperparameters["timing"]:

        print_overloaded("Starting timing, will run 5 times")
        times = []

        for i in range(1,6):
            
            print_overloaded("timing iteration", i)
            testcont = Function(vCG)

            testcont.vector()[:] = i / 100
            t0 = time.time()

            Jhat(testcont)

            dt0 = time.time() - t0

            times.append(dt0)

            print_overloaded("timing iteration", i, "took", dt0 / 3600, "h")
            print_overloaded("Using", MPI.comm_world.Get_size(), "processes")
        
        if MPI.rank(MPI.comm_world) == 0:
            hyperparameters["times"] = times
            hyperparameters["processes"] = MPI.comm_world.Get_size()

            with open(hyperparameters["outputfolder"] + '/hyperparameters.json', 'w') as outfile:
                json.dump(hyperparameters, outfile, sort_keys=True, indent=4)

        print_overloaded("Timing done, exiting")
        exit()

    if not hyperparameters["memdebug"]:

        files["stateFile"].write(Img_deformed, str(0))
        files["controlFile"].write(l2_controlfun, str(0))

    if hyperparameters["debug"]:

        print_overloaded("Running convergence test")
        h = Function(vCG)
        h.vector()[:] = 0.1
        h.vector().apply("")
        conv_rate = taylor_test(Jhat, velocity, h)
        print_overloaded(conv_rate)
        print_overloaded("convergence test done, exiting")
        hyperparameters["conv_rate"] = float(conv_rate)  
        exit()
        



    def cb(*args, **kwargs):
        global current_iteration
        current_iteration += 1
        
        mem = resource.getrusage(resource.RUSAGE_SELF)[2]
        print("Memory (TB)", (mem/(1e6*1024)), "current_iteration", current_iteration, "process", str(MPI.rank(MPI.comm_world)))



        # return 

        with stop_annotating():
            current_pde_solution = state.tape_value()
            current_pde_solution.rename("Img", "")
            current_l2_control = cont.tape_value()
            current_pde_solution.vector().update_ghost_values()
            current_l2_control.vector().update_ghost_values()
            current_l2_control.rename("control", "")


            if hyperparameters["memdebug"]:
                return

            if not hyperparameters["memdebug"] and (current_pde_solution.vector()[:].max() > hyperparameters["max_voxel_intensity"] * 5):
                raise ValueError("State became > hyperparameters['max_voxel_intensity'] * 5 at some vertex, something is probably wrong")


            l2loss = (current_pde_solution - Img_goal) ** 2
            l2loss = assemble(0.5 * l2loss * dx)


            if hyperparameters["tukey"]:
                loss = tukeyloss(x=current_pde_solution, y=Img_goal, hyperparameters=hyperparameters)
                Jd = assemble(0.5 * loss * dx)
    
            else:
                Jd = l2loss

            # if hyperparameters["smoothen"]:
            #     current_L2_control = transformation(current_l2_control, M_lumped_inv)
            # else:
            #     current_L2_control = current_l2_control        
            
            # velocityField = preconditioning(current_L2_control)
            # velocityField.rename("velocity", "")

            # # Jd = assemble(0.5 * (current_pde_solution - Img_goal)**2 * dx(domain=Img.function_space().mesh()))
            # Jreg = assemble(alpha*current_L2_control**2*dx(domain=Img.function_space().mesh()))

            domainmesh = current_pde_solution.function_space().mesh()
            
            #compute CFL number
            # h = CellDiameter(domainmesh)
            
            # if hyperparameters["projector"]:
            #     CFL = projectorU.project(sqrt(inner(velocityField, velocityField))*Constant(hyperparameters["DeltaT"]) / h)
            
            # else:
            #     CFL = project(sqrt(inner(velocityField, velocityField))*Constant(hyperparameters["DeltaT"]) / h, FunctionSpace(domainmesh, "DG", 0))

            # if(CFL.vector().max() > 1.0):
                
            #     raise CFLerror("DGTransport: WARNING: CFL = %le", CFL)
                    

            mem = resource.getrusage(resource.RUSAGE_SELF)[2]
            print("Memory (TB)", (mem/(1e6*1024)), "current_iteration", current_iteration + 0.5, "process", str(MPI.rank(MPI.comm_world)))


            store_during_callback(current_iteration, hyperparameters, files, Jd, l2loss,
                                        domainmesh, current_pde_solution, control=current_l2_control)


    print_overloaded("Using maxcor =", hyperparameters["maxcor"], "in LBFGS-B")

    t0 = time.time()

    if hyperparameters["memdebug"]:
        tol = 1e-32

    else:
        tol = 1e-8

    minimize(Jhat,  method = 'L-BFGS-B', options = {"iprint": 0, "disp": None, "maxiter": hyperparameters["lbfgs_max_iterations"],
                # "maxls": 1,  "ftol": 0, "gtol": 0, 
                "maxcor": hyperparameters["maxcor"]}, tol=tol, callback = cb)


    # minimize(Jhat,  method = 'CG', options = {"disp": False, "maxiter": hyperparameters["lbfgs_max_iterations"], "gtol":tol}, tol=tol, callback = cb)

    dt0 = time.time() - t0

    hyperparameters["minimization_time_hours"] = dt0 / 3600


    if MPI.rank(MPI.comm_world) == 0:
        with open(hyperparameters["outputfolder"] + '/hyperparameters_after_min.json', 'w') as outfile:
            json.dump(hyperparameters, outfile, sort_keys=True, indent=4)


    if hyperparameters["memdebug"]:
        exit()

    current_pde_solution = state.tape_value()
    current_pde_solution.rename("finalstate", "")
    # File(hyperparameters["outputfolder"] + '/Finalstate.pvd') << current_pde_solution

    final_l2_control = cont.tape_value()
    final_l2_control.rename("control", "")

    if hyperparameters["smoothen"]:
        scaledControl = transformation(final_l2_control, M_lumped_inv)
    else:
        scaledControl = final_l2_control

    velocityField = preconditioning(scaledControl)
    velocityField.rename("velocity", "")

    return current_pde_solution, velocityField, final_l2_control