from fenics import *
from fenics_adjoint import *

# from mri_utils.helpers import load_velocity, interpolate_velocity
from DGTransport import Transport
from transformation_overloaded import transformation
# from preconditioning_overloaded import Overloaded_Preconditioning # 
from preconditioning_overloaded import preconditioning

class CFLerror(ValueError):
    '''raise this when CFL is violated'''

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass

# current_iteration = 0

def find_velocity(Img, Img_goal, vCG, M_lumped, hyperparameters, files, starting_guess):

    # import config

    # print_overloaded("config.hyper", sorted(config.hyperparameters))
    # print_overloaded("hyper", sorted(hyperparameters))
    # preconditioning = Overloaded_Preconditioning(hyperparameters)

    set_working_tape(Tape())

    # initialize control
    controlfun = Function(vCG)

    # if hyperparameters["interpolate"]:
    #     interpolate_velocity(hyperparameters, controlfun)
    #     exit()

    if hyperparameters["vinit"] > 0:
        controlfun.vector()[:] += hyperparameters["vinit"]
        print_overloaded("----------------------------------- Setting v += vinit as starting guess")

    if hyperparameters["starting_guess"] is not None:
        # load_velocity(hyperparameters, controlfun=controlfun)
        controlfun.assign(starting_guess)

    # raise NotImplementedError()


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

    J = assemble(0.5 * (Img_deformed - Img_goal) ** 2 * dx(domain=Img.function_space().mesh()))
    print_overloaded("Assembled L2 error between transported image and target, Jdata=", J)
    # print_overloaded("Control type=", type(control))
    # print_overloaded(control)
    # J = J + assemble(alpha*grad(control)**2*dx(domain=Img.function_space().mesh()))
    J = J + assemble(alpha*(controlf)**2*dx(domain=Img.function_space().mesh()))
    print_overloaded("Assembled regularization, J=", J)

    Jhat = ReducedFunctional(J, cont)

    print_overloaded("created reduced functional")

    # current_iteration = 0

    files["stateFile"].write(Img_deformed, str(0))
    files["controlFile"].write(control, str(0))

    print_overloaded("Wrote fCont0 to file")    

    # if hyperparameters["debug"]:

    #     print_overloaded("Running convergence test")
    #     h = Function(vCG)
    #     h.vector()[:] = 0.1
    #     h.vector().apply("")
    #     conv_rate = taylor_test(Jhat, control, h)
    #     print_overloaded(conv_rate)
    #     print_overloaded("convergence test done, exiting")
        
    #     hyperparameters["conv_rate"] = float(conv_rate)
    #     return


    def cb(*args, **kwargs):
        # global current_iteration
        # current_iteration += 1


        current_pde_solution = state.tape_value()
        current_pde_solution.rename("Img", "")
        current_control = cont.tape_value()
        current_control.rename("control", "")

        if current_pde_solution.vector()[:].max() > 10:
            raise ValueError("State became > 10 at some vertex, something is probably wrong")

        Jd = assemble(0.5 * (current_pde_solution - Img_goal)**2 * dx(domain=Img.function_space().mesh()))
        Jreg = assemble(alpha*(current_control)**2*dx(domain=Img.function_space().mesh()))

        if MPI.rank(MPI.comm_world) == 0:
            
            hyperparameters["Jd"].append(float(Jd))
            hyperparameters["Jreg"].append(float(Jreg))

            files["lossfile"].write(str(float(Jd))+ ", ")
            files["regularizationfile"].write(str(float(Jreg))+ ", ")

        print_overloaded("J=", Jd, "Reg=", Jreg)

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
        
        files["velocityFile"].write(velocityField, "-1") # str(current_iteration))
        files["controlFile"].write(current_control, "-1") #str(current_iteration))
        files["stateFile"].write(current_pde_solution, "-1") # str(current_iteration))

        File(hyperparameters["outputfolder"] + '/Currentstate.pvd') << current_pde_solution

        print_overloaded("Wrote files in callback")

    minimize(Jhat,  method = 'L-BFGS-B', options = {"disp": True, "maxiter": hyperparameters["lbfgs_max_iterations"]}, tol=1e-08, callback = cb)

    # Store final values in pvd format for visualization

    current_pde_solution = state.tape_value()
    current_pde_solution.rename("finalstate", "")

    File(hyperparameters["outputfolder"] + '/Finalstate.pvd') << current_pde_solution

    current_control = cont.tape_value()
    current_control.rename("control", "")

    if hyperparameters["smoothen"]:
        scaledControl = transformation(current_control, M_lumped)

    else:
        scaledControl = current_control

    velocityField = preconditioning(scaledControl)
    velocityField.rename("velocity", "")

    File(hyperparameters["outputfolder"] + '/Finalvelocity.pvd') << velocityField
    File(hyperparameters["outputfolder"] + '/Finalcontrol.pvd') << current_control

    print_overloaded("Stored final State, Control, Velocity to .pvd files")