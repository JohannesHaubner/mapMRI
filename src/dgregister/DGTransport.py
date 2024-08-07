#solver for transporting images
from fenics import *
from fenics_adjoint import *
import numpy as np
from dgregister.MRI2FEM import fem2mri
import os, pathlib
from dgregister.helpers import crop_to_original
import nibabel


def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass


print_overloaded("Setting parameters parameters['ghost_mode'] = 'shared_facet'")
parameters['ghost_mode'] = 'shared_facet'


q_degree = 6


def DGTransport(Img, Wind, MaxIter, DeltaT, timestepping, solver="krylov", preconditioner="amg", MassConservation=False, 
                storage_info=None):
    
    print_overloaded("......................................")
    print_overloaded("Settings in Transport()")
    print_overloaded("--- timestepping =", timestepping)
    print_overloaded("......................................")

    print_overloaded("parameters['ghost_mode']", parameters['ghost_mode'])

    
    Space = Img.function_space()
    v = TestFunction(Space)

    mesh = Space.mesh()

    #Make form:
    n = FacetNormal(mesh)
    def Max0(d):
        return 0.5*(d+abs(d))
    
    def Max0smoothed(d, k=1):
        return d/(1+exp(-Constant(k)*d))

    #Scheme = "Central" #needed for Taylor-Test
    #Scheme = "Upwind"
    Scheme = "smoothedUpwind"
    
    def Flux(f, Wind, n):
        if Scheme == "Central":
            flux = 0.5*inner(Wind, n)
        if Scheme == "Upwind":
            flux = Max0(inner(Wind,n))
        if Scheme == "smoothedUpwind":
            flux = Max0smoothed(inner(Wind, n))
        return f*flux
        
    def FluxB(f, Wind, n):
        if Scheme == "Central":
            return f*inner(Wind,n)
        if Scheme == "Upwind":
            return f*Max0(inner(Wind,n))
        if Scheme == "smoothedUpwind":
            return f*Max0smoothed(inner(Wind, n))


    def Form(f):   
        a = -inner(grad(v), outer(f, Wind)) * dx
        a += inner(jump(v), jump(Flux(f, Wind, n))) * dS(metadata={'quadrature_degree': q_degree})
        
        # The following term is 0 for the boundary condition v = 0
        # included for general BC for v
        a += inner(v, FluxB(f, Wind, n)) * ds(metadata={'quadrature_degree': q_degree})
    
        if MassConservation == False:
            a -= inner(v, div(Wind)*f)*dx
        return a

    Img_next = TrialFunction(Img.function_space())
    Img_deformed = Function(Img.function_space())
    Img_deformed.assign(Img)
    Img_deformed.rename("Img", "")

    a = Constant(1.0/DeltaT)*(inner(v,Img_next)*dx - inner(v, Img_deformed)*dx)
    
    if timestepping == "explicitEuler":
        a = a + Form(Img_deformed)
    
    elif timestepping == "RungeKutta":
        # in this case we assemble the RHS during the loop
        dImg = TrialFunction(Img_deformed.function_space())
        dI = Function(Img_deformed.function_space())

        form = inner(dImg, v)*dx 
        Atmp = assemble(form)
        tmpsolver = KrylovSolver(method="cg", preconditioner=preconditioner)
        tmpsolver.set_operators(Atmp, Atmp)

    elif timestepping == "CrankNicolson":
        a = a + 0.5*(Form(Img_deformed) + Form(Img_next))
    
    else:
        raise NotImplementedError

    A = assemble(lhs(a))

    if solver == "krylov":
        solver = KrylovSolver(A, "gmres", preconditioner)
        solver.set_operators(A, A)
        print_overloaded("Assembled A, using Krylov solver")
    
    elif solver == "lu":
        solver = LUSolver()
        solver.set_operator(A)
        print_overloaded("Assembled A, using LU solver")

    elif solver == "cg":
        solver = KrylovSolver(method="cg", preconditioner=preconditioner)
        solver.set_operators(A, A)
        print_overloaded("Assembled A, using CG solver")

    else:
        raise NotImplementedError()

    b = None
    btmp = None

    for i in range(MaxIter):

        print_overloaded("Iteration ", i + 1, "/", MaxIter, "in Transport()")

        if timestepping == "RungeKutta":
            
            rhstmp = Form(Img_deformed)            
            
            if btmp is None:
                
                btmp = assemble(rhstmp)
            else:
                btmp = assemble(rhstmp, tensor=btmp)

            btmp.apply("")

            tmpsolver.solve(dI.vector(), btmp)

            factor = DeltaT / 2.

            da = Form(Img_deformed + factor * dI)
            
            system_rhs = rhs(a + da)
        else:
            system_rhs = rhs(a)

        if b is None:
            b = assemble(system_rhs)
        else:
            b = assemble(system_rhs, tensor=b)
        solver.solve(Img.vector(), b)

        Img_deformed.assign(Img)

        if storage_info is not None:
            """
            Useful for storing state during transport for visualization
            """
            store_states_to = pathlib.Path(storage_info["store_states_to"])

            assert os.path.isdir(store_states_to)

            retimage = fem2mri(function=Img_deformed, shape=storage_info["shape"])

            storepath = store_states_to / ("state" + format(i + 1, ".0f") + ".npy")

            np.save(storepath, retimage)

            filled_image = crop_to_original(orig_image=np.zeros((256, 256, 256)), 
                                            cropped_image=retimage, box=storage_info["box"], 
                                            space=storage_info["space"], pad=storage_info["pad"])

            nii = nibabel.Nifti1Image(filled_image, storage_info["aff"])
            nibabel.save(nii, str(storepath).replace(".npy", ".mgz"))

            Img_deformed.rename("state", "state")
            storage_info["pvdfile"] << Img_deformed

            print_overloaded("-- stored", str(storepath))

    print_overloaded("i == MaxIter, Transport() finished")

    return Img_deformed












if __name__ == "__main__":
    #create on the fly
    FName = "shuttle_small.png"
    from dgregister.Pic2Fen import Pic2FEM, FEM2Pic
    (mesh, Img, NumData) = Pic2FEM(FName)
    
    FNameOut = "img_DG"
    FNameOut = "output/"+FNameOut+".xdmf"
    StoreHistory = True
    MassConservation = False
    MaxIter = 300
    DeltaT = 2e-4
    
    x = SpatialCoordinate(mesh)
    Wind = as_vector((0.0, x[1]))

    Img = project(Img, VectorFunctionSpace(mesh, "DG", 1, NumData))
    Img.rename("img", "")

    Img_deformed = DGTransport(Img, Wind, MaxIter, DeltaT, MassConservation, StoreHistory, FNameOut)
    File("output/DGTransportFinal.pvd") << Img_deformed
    FEM2Pic(Img_deformed, NumData, "output/DGTransportFinal.png")

