from fenics import *
from fenics_adjoint import *

# from mri_utils.helpers import load_velocity, interpolate_velocity
from dgregister.DGTransport import DGTransport
from dgregister.transformation_overloaded import transformation
# from preconditioning_overloaded import Overloaded_Preconditioning # 
from dgregister.preconditioning_overloaded import preconditioning

import time, json
# import nibabel
# from dgregister.MRI2FEM import fem2mri

from dgregister.helpers import store_during_callback


set_log_level(LogLevel.CRITICAL)


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

def find_velocity(Img, Img_goal, vCG, M_lumped_inv, hyperparameters, files, starting_guess):

    set_working_tape(Tape())

    # initialize control
    controlfun = Function(vCG)

    if (hyperparameters["starting_guess"] is not None) and (not hyperparameters["interpolate"]):
        controlfun.assign(starting_guess)

        assert norm(controlfun) > 0


    if hyperparameters["smoothen"]:
        print_overloaded("Transforming l2 control to L2 control")
        controlf = transformation(controlfun, M_lumped_inv)
    else:
        controlf = controlfun

    control = preconditioning(controlf)

    control.rename("control", "")

    print_overloaded("Running Transport() with dt = ", hyperparameters["DeltaT"])


    if hyperparameters["interpolate"]:
        print_overloaded("Start transporting with starting guess, now transport with the new control in addition")
        Img_deformed = DGTransport(Img, Wind=starting_guess, hyperparameters=hyperparameters,
                                MaxIter=hyperparameters["max_timesteps"], DeltaT=hyperparameters["DeltaT"], timestepping=hyperparameters["timestepping"], 
                                solver=hyperparameters["solver"], MassConservation=hyperparameters["MassConservation"])

        print_overloaded("Done transporting with starting guess, now transport with the new control in addition")
        print_overloaded("*"*80)
        Img_deformed = DGTransport(Img_deformed, Wind=control, hyperparameters=hyperparameters,
                                MaxIter=hyperparameters["max_timesteps"], DeltaT=hyperparameters["DeltaT"], timestepping=hyperparameters["timestepping"], 
                                solver=hyperparameters["solver"], MassConservation=hyperparameters["MassConservation"])

    else:

        Img_deformed = DGTransport(Img, control, hyperparameters=hyperparameters,
                                MaxIter=hyperparameters["max_timesteps"], DeltaT=hyperparameters["DeltaT"], timestepping=hyperparameters["timestepping"], 
                                solver=hyperparameters["solver"], MassConservation=hyperparameters["MassConservation"])


    # solve forward and evaluate objective
    alpha = Constant(hyperparameters["alpha"]) #regularization

    state = Control(Img_deformed)  # The Control type enables easy access to tape values after replays.
    cont = Control(controlfun)

    # Jd = assemble(0.5 * (Img_deformed - Img_goal) ** 2 * dx(domain=Img.function_space().mesh()))
    Jd = assemble(0.5 * (Img_deformed - Img_goal) ** 2 * dx) # (domain=Img_goal.function_space().mesh()))
    print_overloaded("Assembled L2 error between transported image and target, Jdata=", Jd)

    Jreg = assemble(alpha*(controlf)**2*dx) # (domain=Img.function_space().mesh()))
    
    J = Jd + Jreg
    
    hyperparameters["Jd_init"] = float(Jd)
    hyperparameters["Jreg_init"] = float(Jreg)
    
    print_overloaded("Assembled functional, J=", J)

    Jhat = ReducedFunctional(J, cont)

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

    files["stateFile"].write(Img_deformed, str(0))
    files["controlFile"].write(control, str(0))

    if hyperparameters["debug"]:

        print_overloaded("Running convergence test")
        h = Function(vCG)
        h.vector()[:] = 0.1
        h.vector().apply("")
        conv_rate = taylor_test(Jhat, control, h)
        print_overloaded(conv_rate)
        print_overloaded("convergence test done, exiting")
        
        hyperparameters["conv_rate"] = float(conv_rate)
        return



    def cb(*args, **kwargs):
        global current_iteration
        current_iteration += 1

        current_pde_solution = state.tape_value()
        current_pde_solution.rename("Img", "")
        current_control = cont.tape_value()
        current_pde_solution.vector().update_ghost_values()
        current_control.vector().update_ghost_values()
        current_control.rename("control", "")

        if current_pde_solution.vector()[:].max() > 10:
            raise ValueError("State became > 10 at some vertex, something is probably wrong")

        Jd = assemble(0.5 * (current_pde_solution - Img_goal)**2 * dx(domain=Img.function_space().mesh()))
        Jreg = assemble(alpha*(current_control)**2*dx(domain=Img.function_space().mesh()))

        domainmesh = current_pde_solution.function_space().mesh()
        
        #compute CFL number
        h = CellDiameter(domainmesh)
        CFL = project(sqrt(inner(current_control, current_control))*Constant(hyperparameters["DeltaT"]) / h, FunctionSpace(domainmesh, "DG", 0))

        if(CFL.vector().max() > 1.0):
            
            raise CFLerror("DGTransport: WARNING: CFL = %le", CFL)
                
        if hyperparameters["smoothen"]:
            controlfunction = current_control
            scaledControl = transformation(current_control, M_lumped_inv)
        else:
            scaledControl = current_control
            controlfunction = None

        velocityField = preconditioning(scaledControl)
        velocityField.rename("velocity", "")

        store_during_callback(current_iteration=current_iteration, hyperparameters=hyperparameters, files=files, Jd=Jd, Jreg=Jreg, 
                            domainmesh=domainmesh, velocityField=velocityField, 
                            current_pde_solution=current_pde_solution, control=controlfunction)


    minimize(Jhat,  method = 'L-BFGS-B', options = {"iprint": 0, "disp": None, "maxiter": hyperparameters["lbfgs_max_iterations"]}, tol=1e-08, callback = cb)



    current_pde_solution = state.tape_value()
    current_pde_solution.rename("finalstate", "")
    # File(hyperparameters["outputfolder"] + '/Finalstate.pvd') << current_pde_solution

    current_control = cont.tape_value()
    current_control.rename("control", "")

    if hyperparameters["smoothen"]:
        scaledControl = transformation(current_control, M_lumped_inv)
        returncontrol = current_control
    else:
        scaledControl = current_control
        returncontrol = None

    velocityField = preconditioning(scaledControl)
    velocityField.rename("velocity", "")

    return current_pde_solution, velocityField, returncontrol