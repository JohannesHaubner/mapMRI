from fenics import *
from fenics_adjoint import *

# from mri_utils.helpers import load_velocity, interpolate_velocity
from DGTransport import Transport
from transformation_overloaded import transformation
# from preconditioning_overloaded import Overloaded_Preconditioning # 
from preconditioning_overloaded import preconditioning

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

def find_velocity(Img, Img_goal, vCG, M_lumped, hyperparameters, files, starting_guess):

    set_working_tape(Tape())

    # initialize control
    controlfun = Function(vCG)

    if hyperparameters["vinit"] > 0:
        controlfun.vector()[:] += hyperparameters["vinit"]
        print_overloaded("----------------------------------- Setting v += vinit as starting guess")

    if hyperparameters["starting_guess"] is not None:
        controlfun.assign(starting_guess)


    if hyperparameters["smoothen"]:
        print_overloaded("Transforming l2 control to L2 control")
        controlf = transformation(controlfun, M_lumped)
    else:
        controlf = controlfun

    control = preconditioning(controlf)

    control.rename("control", "")

    print_overloaded("Running Transport() with dt = ", hyperparameters["DeltaT"])

    Img_deformed = Transport(Img, control, hyperparameters=hyperparameters,
                            MaxIter=hyperparameters["max_timesteps"], DeltaT=hyperparameters["DeltaT"], timestepping=hyperparameters["timestepping"], 
                            solver=hyperparameters["solver"], MassConservation=hyperparameters["MassConservation"])

    # solve forward and evaluate objective
    alpha = Constant(hyperparameters["alpha"]) #regularization

    state = Control(Img_deformed)  # The Control type enables easy access to tape values after replays.
    cont = Control(controlfun)

    Jd = assemble(0.5 * (Img_deformed - Img_goal) ** 2 * dx(domain=Img.function_space().mesh()))
    print_overloaded("Assembled L2 error between transported image and target, Jdata=", Jd)

    Jreg = assemble(alpha*(controlf)**2*dx(domain=Img.function_space().mesh()))
    
    J = Jd + Jreg
    
    hyperparameters["Jd_init"] = float(Jd)
    hyperparameters["Jreg_init"] = float(Jreg)
    
    print_overloaded("Assembled functional, J=", J)

    Jhat = ReducedFunctional(J, cont)

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

        if MPI.rank(MPI.comm_world) == 0:
       
            with open(files["lossfile"], "a") as myfile:
                myfile.write(str(float(Jd))+ ", ")
            with open(files["regularizationfile"], "a") as myfile:
                myfile.write(str(float(Jreg))+ ", ")
        
        hyperparameters["Jd_current"] = float(Jd)
        hyperparameters["Jreg_current"] = float(Jreg)
        
        print_overloaded("Iter", format(current_iteration, ".0f"), "Jd =", format(Jd, ".2e"), "Reg =", format(Jreg, ".2e"))

        domainmesh = current_pde_solution.function_space().mesh()
        
        #compute CFL number
        h = CellDiameter(domainmesh)
        CFL = project(sqrt(inner(current_control, current_control))*Constant(hyperparameters["DeltaT"]) / h, FunctionSpace(domainmesh, "DG", 0))

        if(CFL.vector().max() > 1.0):
            
            raise CFLerror("DGTransport: WARNING: CFL = %le", CFL)
                
        if hyperparameters["smoothen"]:
            scaledControl = transformation(current_control, M_lumped)
        else:
            scaledControl = current_control

        velocityField = preconditioning(scaledControl)
        velocityField.rename("velocity", "")

        # # This does not overwrite the files (bug warning supressed due to import of h5py)        
        # files["velocityFile"].write(velocityField, "-1") # str(current_iteration))
        # files["controlFile"].write(current_control, "-1") #str(current_iteration))
        # files["stateFile"].write(current_pde_solution, "-1") # str(current_iteration))

        # File(hyperparameters["outputfolder"] + '/Currentstate.pvd') << current_pde_solution

        with XDMFFile(hyperparameters["outputfolder"] + "/State_checkpoint.xdmf") as xdmf:
            xdmf.write_checkpoint(current_pde_solution, "CurrentState", 0.)
        with XDMFFile(hyperparameters["outputfolder"] + "/Velocity_checkpoint.xdmf") as xdmf:
            xdmf.write_checkpoint(velocityField, "CurrentV", 0.)


    minimize(Jhat,  method = 'L-BFGS-B', options = {"iprint": 0, "disp": None, "maxiter": hyperparameters["lbfgs_max_iterations"]}, tol=1e-08, callback = cb)

    hyperparameters["Jd_final"] = hyperparameters["Jd_current"]
    hyperparameters["Jreg_final"] = hyperparameters["Jreg_current"]


    current_pde_solution = state.tape_value()
    current_pde_solution.rename("finalstate", "")
    # File(hyperparameters["outputfolder"] + '/Finalstate.pvd') << current_pde_solution

    current_control = cont.tape_value()
    current_control.rename("control", "")

    if hyperparameters["smoothen"]:
        scaledControl = transformation(current_control, M_lumped)

    else:
        scaledControl = current_control

    velocityField = preconditioning(scaledControl)
    velocityField.rename("velocity", "")

    # File(hyperparameters["outputfolder"] + '/Finalvelocity.pvd') << velocityField
    # File(hyperparameters["outputfolder"] + '/Finalcontrol.pvd') << current_control

    with XDMFFile(hyperparameters["outputfolder"] + "/Finalstate.xdmf") as xdmf:
        xdmf.write_checkpoint(current_pde_solution, "Finalstate", 0.)
    
    with XDMFFile(hyperparameters["outputfolder"] + "/Finalvelocity.xdmf") as xdmf:
        xdmf.write_checkpoint(velocityField, "FinalV", 0.)

    files["velocityFile"].write(velocityField, "-1")
    files["controlFile"].write(current_control, "-1")
    files["stateFile"].write(current_pde_solution, "-1")

    print_overloaded("Stored final State, Control, Velocity to .hdf files")
