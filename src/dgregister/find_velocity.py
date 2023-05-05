from fenics import *
from fenics_adjoint import *
from dgregister.DGTransport import DGTransport
from dgregister.transformation_overloaded import transformation
from dgregister.preconditioning_overloaded import preconditioning
from dgregister.huber import huber
import time, json
import numpy as np
import resource
from dgregister.helpers import store_during_callback



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

def find_velocity(starting_image, Img_goal, vCG, M_lumped_inv, hyperparameters, files, starting_guess=None, storage_info=None):

    vol = assemble(1*dx(starting_image.function_space().mesh()))

    hyperparameters["vol"] = vol

    set_working_tape(Tape())

    # initialize control
    l2_controlfun = Function(vCG)

    if (hyperparameters["starting_guess"] is not None):
        l2_controlfun.assign(starting_guess)
        print_overloaded("*"*20, "assigned starting guess to control")
        assert norm(l2_controlfun) > 0
    
    control_L2 = transformation(l2_controlfun, M_lumped_inv)
    print_overloaded("Preconditioning L2_controlfun, name=", control_L2)
    velocity = preconditioning(control_L2)

    l2_controlfun.rename("control_l2", "")
    control_L2.rename("control_L2", "")
    velocity.rename("velocity", "")

    print_overloaded("Running Transport() with dt = ", hyperparameters["DeltaT"])

    Img_deformed = DGTransport(starting_image, velocity, MaxIter=hyperparameters["max_timesteps"], DeltaT=hyperparameters["DeltaT"], timestepping=hyperparameters["timestepping"], 
                            MassConservation=hyperparameters["MassConservation"], storage_info=storage_info)

    if hyperparameters["forward"]:

        print_overloaded("*" * 100)
        print_overloaded("--forward is set, run forward simulation. Returning deformed image and not optimizing.")
        print_overloaded("*" * 100)

        return Img_deformed, velocity, l2_controlfun

    if storage_info is not None:
        print_overloaded("*" * 100)
        print_overloaded("Stored state at all timesteps, exiting script")
        print_overloaded("*" * 100)
        exit()

    alpha = Constant(hyperparameters["alpha"]) # regularization
    state = Control(Img_deformed)  # The Control type enables easy access to tape values after replays.
    cont = Control(l2_controlfun)


    if hyperparameters["tukey"]:

        def tukeyloss(x, y, hyperparameters):

            tukey_c = hyperparameters["tukey_c"]

            residual = x - y

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

        with stop_annotating():
            l2loss = assemble(0.5 * (Img_deformed - Img_goal) ** 2 * dx)

    elif hyperparameters["huber"]:
        def huberloss(x, y, hyperparameters):

            delta = hyperparameters["huber_delta"]

            residual = x - y

            with stop_annotating():
                arrname = hyperparameters["outputfolder"] + "/normalized_residual" + str(MPI.rank(MPI.comm_world)) + ".npy"
                resvec = x.vector()[:] - y.vector()[:] #  - mean_residual) / std_residual
                np.save(arrname, resvec)

                print_overloaded("Stored residuals")
            
            loss = huber(x=residual, delta=delta)
            
            return loss

        print_overloaded("Using huber loss")
        loss = huberloss(x=Img_deformed, y=Img_goal, hyperparameters=hyperparameters)
        Jd = assemble(loss * dx)

        with stop_annotating():
            l2loss = assemble(0.5 * (Img_deformed - Img_goal) ** 2 * dx)

    else:
        Jd = assemble(0.5 * (Img_deformed - Img_goal) ** 2 * dx)
        
        with stop_annotating():
            l2loss = Jd

    assert Jd > 0    

    print_overloaded("Assembled error between transported image and target, Jdata=", Jd)
    print_overloaded("L2 error between transported image and target, Jdata_L2=", l2loss)

    Jreg = assemble(alpha*(control_L2)**2*dx)
    
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


    # files["stateFile"].write(Img_deformed, str(0))
    # files["controlFile"].write(l2_controlfun, str(0))

    if hyperparameters["taylortest"]:

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
        
        if hyperparameters["memdebug"]:
            mem = resource.getrusage(resource.RUSAGE_SELF)[2]
            print("Memory (TB)", (mem/(1e6*1024)), "current_iteration", current_iteration, "process", str(MPI.rank(MPI.comm_world)))


        with stop_annotating():
            current_pde_solution = state.tape_value()
            current_pde_solution.rename("Img", "")
            current_l2_control = cont.tape_value()
            current_pde_solution.vector().update_ghost_values()
            current_l2_control.vector().update_ghost_values()
            current_l2_control.rename("control", "")


            if (current_pde_solution.vector()[:].max() > hyperparameters["max_voxel_intensity"] * 5):
                print_overloaded("*"*80)
                print_overloaded("State became > hyperparameters['max_voxel_intensity'] * 5 at some vertex, something is probably wrong")
                print_overloaded("*"*80)


            l2loss = (current_pde_solution - Img_goal) ** 2
            l2loss = assemble(0.5 * l2loss * dx)

            if hyperparameters["tukey"]:
                loss = tukeyloss(x=current_pde_solution, y=Img_goal, hyperparameters=hyperparameters)
                Jd = assemble(0.5 * loss * dx)
            elif hyperparameters["huber"]:
                loss = huberloss(x=current_pde_solution, y=Img_goal, hyperparameters=hyperparameters)
                Jd = assemble(loss * dx)
            else:
                Jd = l2loss

            domainmesh = current_pde_solution.function_space().mesh()

            # def store_during_callback(current_iteration, hyperparameters, files, Jd, l2loss,
            #                domainmesh, current_pde_solution, control):
            store_during_callback(current_iteration, hyperparameters, files, Jd, l2loss,
                                        domainmesh, current_pde_solution, control=current_l2_control)


    print_overloaded("Using maxcor =", hyperparameters["maxcor"], "in LBFGS-B")

    t0 = time.time()

    if hyperparameters["memdebug"]:
        tol = 1e-32
    else:
        tol = 1e-8

    minimize(Jhat,  method = 'L-BFGS-B', options = {"iprint": 0, "disp": None, "maxiter": hyperparameters["lbfgs_max_iterations"],
                "maxcor": hyperparameters["maxcor"]}, tol=tol, callback = cb)


    if hyperparameters["memdebug"]:
        exit()

    dt0 = time.time() - t0

    hyperparameters["minimization_time_hours"] = dt0 / 3600


    if MPI.rank(MPI.comm_world) == 0:
        with open(hyperparameters["outputfolder"] + '/hyperparameters_after_min.json', 'w') as outfile:
            json.dump(hyperparameters, outfile, sort_keys=True, indent=4)

    current_pde_solution = state.tape_value()
    current_pde_solution.rename("finalstate", "")

    final_l2_control = cont.tape_value()
    final_l2_control.rename("control", "")

    scaledControl = transformation(final_l2_control, M_lumped_inv)
    velocityField = preconditioning(scaledControl)
    velocityField.rename("velocity", "")


    
    return current_pde_solution, velocityField, final_l2_control