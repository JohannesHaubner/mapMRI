from fenics import *
from fenics_adjoint import *
import os

def print_overloaded(*args):
    if MPI.rank(MPI.comm_world) == 0:
        # set_log_level(PROGRESS)
        print(*args)
    else:
        pass
        # print("passed")

print_overloaded("Setting parameters parameters['ghost_mode'] = 'shared_facet'")
parameters['ghost_mode'] = 'shared_facet'

hyperparameters = {}
# outpath = "/home/bastian/D1/registration/iotest3d/"
outpath = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/iotest/"

hyperparameters["outputfolder"] = outpath + "parallel" # + "Project"
# hyperparameters["outputfolder"] = outpath + "parallel" + "Pic2FEN"
# hyperparameters["outputfolder"] = outpath + "sequential" # + "Project"

if os.path.isdir(hyperparameters["outputfolder"]):
    os.system("rm -r " + hyperparameters["outputfolder"])
else:
    # if MPI.rank(MPI.comm_world) == 0:
    #     os.makedirs(hyperparameters["outputfolder"])
    # else:
    #     pass
    os.makedirs(hyperparameters["outputfolder"], exist_ok=True)

if True not in [x in hyperparameters["outputfolder"] for x in ["MRI2FEM", "Pic2FEN"]]: 

    domainmesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    V1 = FunctionSpace(domainmesh, "DG", 1)
    V0 = FunctionSpace(domainmesh, "DG", 0)

    if "Project" in hyperparameters["outputfolder"]:
        Img0 = interpolate(Expression("x[0] <= 0.5 ? 1 : 0", degree=1), V0)
        Img = project(Img0, V1)
        print_overloaded("Projecting")
    else:
        Img = interpolate(Expression("x[0] <= 0.5 ? 1 : 0", degree=1), V1)
        print_overloaded("Directly interpolate to V1")

else:

    FName = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/testdata_3d/input.mgz"

    if "MRI2FEM" in hyperparameters["outputfolder"]:
        from mri_utils.MRI2FEM import read_image

        normalize = False

        domainmesh, Img, N1 = read_image(hyperparameters={"image": FName, "state_functionspace": "DG", "state_functiondegree":1}, 
                        name="image", mesh=None, normalize=normalize)

    elif "Pic2FEN" in hyperparameters["outputfolder"]:
        
        import Pic2Fen

        domainmesh, img1, N1 = Pic2Fen.Pic2FEM(FName, mesh=None, degree=0)

        V1 = FunctionSpace(domainmesh, "DG", 1)
        Img = project(sqrt(inner(img1, img1)), V1)

breakpoint()

velocityFile = HDF5File(MPI.comm_world, hyperparameters["outputfolder"] + "/Imgh5.hdf", "w")
velocityFile.write(domainmesh, "mesh")
velocityFile.write(Img, "Img")
# velocityFile.parameters["flush_output"] = True
# velocityFile.parameters["rewrite_function_mesh"] = False


file = XDMFFile(MPI.comm_world, hyperparameters["outputfolder"] + "/ImgXdmf.xdmf")
file.parameters["flush_output"] = True
file.parameters["rewrite_function_mesh"] = False
# fCont.write(Img.function_space().mesh(), '/mesh')
file.write(Img, 0)
file.close()

with XDMFFile(hyperparameters["outputfolder"] + "/ImgCheckpoint.xdmf") as xdmf:
    xdmf.write_checkpoint(Img, "checkpoints", 0.)

File(hyperparameters["outputfolder"] + "/ImgPvd.pvd") << Img

exit()