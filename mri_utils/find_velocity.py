from fenics import *
from fenics_adjoint import *

from mri_utils.helpers import load_velocity, interpolate_velocity

from transformation_overloaded import transformation
from preconditioning_overloaded import preconditioning

class CFLerror(ValueError):
    '''raise this when CFL is violated'''


current_iteration = 0

def find_velocity(Img, Img_goal, vCG, M_lumped, hyperparameters, files, starting_guess):
    
    from DGTransport import Transport

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

    File(hyperparameters["outputfolder"] + "/input.pvd") << Img
    File(hyperparameters["outputfolder"] + "/target.pvd") << Img_goal

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

    # if hyperparameters["create_guess_only"]:
    #     try:
    #         control.vector()[:] = 42
    #     except:
    #         pass
    
    files["controlFile"].write(control, str(current_iteration))

    if hyperparameters["create_guess_only"]:

        files["controlFile"].close()
        print("Wrote fCont, close and exit")
        exit()

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
        
        files["velocityFile"].write(velocityField, str(current_iteration))
        files["controlFile"].write(current_control, str(current_iteration))
        files["stateFile"].write(current_pde_solution, str(current_iteration))

    minimize(Jhat,  method = 'L-BFGS-B', options = {"disp": True, "maxiter": hyperparameters["lbfgs_max_iterations"]}, tol=1e-08, callback = cb)
