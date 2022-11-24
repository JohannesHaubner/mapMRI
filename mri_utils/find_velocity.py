from fenics import *
from fenics_adjoint import *

# from mri_utils.helpers import load_velocity, interpolate_velocity
from DGTransport import Transport
from transformation_overloaded import transformation
from preconditioning_overloaded import preconditioning

class CFLerror(ValueError):
    '''raise this when CFL is violated'''


current_iteration = 0

def find_velocity(Img, Img_goal, vCG, M_lumped, hyperparameters, files, starting_guess):
    

    set_working_tape(Tape())

    # initialize control
    controlfun = Function(vCG)

    # if hyperparameters["interpolate"]:
    #     interpolate_velocity(hyperparameters, controlfun)
    #     exit()

    if hyperparameters["starting_guess"] is not None:
        # load_velocity(hyperparameters, controlfun=controlfun)
        controlfun.assign(starting_guess)

    # raise NotImplementedError()


    if hyperparameters["smoothen"]:
        controlf = transformation(controlfun, M_lumped)
    else:
        controlf = controlfun

    control = preconditioning(controlf, smoothen=hyperparameters["smoothen"])

    control.rename("control", "")

    print("Running Transport() with dt = ", hyperparameters["DeltaT"])

    Img_deformed = Transport(Img, control, MaxIter=int(1 / hyperparameters["DeltaT"]), DeltaT=hyperparameters["DeltaT"], timestepping=hyperparameters["timestepping"], 
                            solver=hyperparameters["solver"], MassConservation=hyperparameters["MassConservation"])

    # solve forward and evaluate objective
    alpha = Constant(hyperparameters["alpha"]) #regularization

    state = Control(Img_deformed)  # The Control type enables easy access to tape values after replays.
    cont = Control(controlfun)

    J = assemble(0.5 * (Img_deformed - Img_goal)**2 * dx(domain=Img.function_space().mesh()))
    print("Assembled L2 error between image and target")
    print("Control type=", type(control))
    print(control)
    # J = J + assemble(alpha*grad(control)**2*dx(domain=Img.function_space().mesh()))
    J = J + assemble(alpha*(controlf)**2*dx(domain=Img.function_space().mesh()))
    print("Assembled regularization")

    Jhat = ReducedFunctional(J, cont)

    print("created reduced functional")

    current_iteration = 0

    files["stateFile"].write(Img_deformed, str(current_iteration))
    files["controlFile"].write(control, str(current_iteration))

    print("Wrote fCont0 to file")    

    

    def cb(*args, **kwargs):
        global current_iteration
        current_iteration += 1

        current_pde_solution = state.tape_value()
        current_pde_solution.rename("Img", "")
        current_control = cont.tape_value()
        current_control.rename("control", "")

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

        velocityField = preconditioning(scaledControl, smoothen=hyperparameters["smoothen"])
        velocityField.rename("velocity", "")
        
        files["velocityFile"].write(velocityField, "-1") # str(current_iteration))
        files["controlFile"].write(current_control, "-1") #str(current_iteration))
        files["stateFile"].write(current_pde_solution, "-1") # str(current_iteration))

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

    velocityField = preconditioning(scaledControl, smoothen=hyperparameters["smoothen"])
    velocityField.rename("velocity", "")

    File(hyperparameters["outputfolder"] + '/Finalvelocity.pvd') << velocityField
    File(hyperparameters["outputfolder"] + '/Finalcontrol.pvd') << current_control

    print("Stored final State, Control, Velocity to .pvd files")