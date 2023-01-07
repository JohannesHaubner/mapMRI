import json
from fenics import *
from fenics_adjoint import *
import os
import numpy as np
import matplotlib.pyplot as plt


def get_folders() -> list:
    allfolders = sorted([x for x in os.listdir("./") if os.path.isdir(x)])
    folders = []
    
    for folder in allfolders:

        if "Pic2FEN".lower() in folder.lower():
            continue
        
        try:
            params = json.load(open(folder + "/hyperparameters.json"))
        except FileNotFoundError:
            assert "Finalstate.xdmf" not in os.listdir(folder)
            print(folder, "is missing hyperparameters")
            continue
        
        if "Jd_init" not in params.keys():
            print("Jd_init not in keys for", folder, "slurmid = ", params["slurmid"])
            assert True not in ["log" in x for x in os.listdir(folder)]
            continue

        if folder[0] == "E":
            assert params["timestepping"] == "explicitEuler"

        loss = np.genfromtxt(folder + "/loss.txt", delimiter=",")[:-1]
        
        # print(params["timestepping"])
        
        folders.append([folder, params, loss])

    return folders


def remove_nonconverged(folders: list) -> list:

    retval = []

    plt.figure()

    for folder, params, loss in folders:
        



        if params["lbfgs_max_iterations"] == len(loss):
            
            if len(loss) == 10000:
                print("10000 iterations are counted as converged")

            elif len(loss) == 4000:
                print("4000 iterations are counted as converged")

            else:
                plt.semilogy(loss, label=folder)
                print(folder, "not converged, removing")
                continue

        
        retval.append([folder, params, loss])
    plt.legend()
    # plt.show()

    return retval


#################################################################################################################################
#################################################################################################################################
#################################################################################################################################


os.chdir("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/boxregistration_outputs")

# os.chdir("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/cuberegistration_outputs")

folders = get_folders()

if "boxregistration_outputs" in os.getcwd():
    folders = remove_nonconverged(folders)

fig2, ax2 = plt.subplots()

fig1, ax1 = plt.subplots()

for folder, params, loss in folders:
    
    
    
    if "RK1000" in folder:
        folder2 = folder.replace("1000", "100")

        if os.path.isfile(folder2 + "/loss.txt"):
            loss2 = np.genfromtxt(folder2 + "/loss.txt", delimiter=",")[:-1]
            if not len(loss2) == len(loss):
                print("Optimization for different number of epochs !!!")
                print(len(loss), "vs", len(loss2))
            print("::::::::::::::::::::::::::::::::::::::::::::", folder.replace("RK1000", ""), loss[-1], loss2[-1])

#     domainmesh = Mesh()

#     hdf = HDF5File(domainmesh.mpi_comm(), folder + "/State.hdf", "r")
#     hdf.read(domainmesh, "mesh", False)
#     V0 = FunctionSpace(domainmesh, "DG", 0)
#     V1 = FunctionSpace(domainmesh, "DG", 1)
#     if "DG0" in folder:
#         V = V0
#     else:
#         V = V1
#     # V=V1
    
#     u = Function(V)
#     hdf.read(u, "-1")
#     hdf.close()

#     u = project(u, V1)

#     _, u_target, _ = MRI2FEM.read_image(hyperparameters={"image": "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/testdata_2d/target.mgz", 
#                         "state_functionspace": "DG", "state_functiondegree": 1}, 
#                         name="image", printout=False, mesh=domainmesh)

#     Jd = assemble(0.5 * (u - u_target) ** 2 * dx(domain=domainmesh))

#     print(np.argmin(Jd-loss), loss[np.argmin(Jd-loss)], Jd)
#     print("DG0" in folder, params["Jd_final"], Jd)
# exit()
# for i in []:
    # if len(loss) == 1000:
    #     print(folder)
    #     raise ValueError

    print("-----", folder, "-----")
    print(len(loss), "iterations of LBFGS", format(float(params["optimization_time_hours"]), ".1f"), "hours")

    print(format(params["Jd_init"], ".1e"), "-->", format(params["Jd_final"], ".1e"))
    # print("Read")

    marker="o"
    color="green"
    markersize = 10


    if "DG0" in folder:
        color="blue"
        marker="x"
    if "NOSMOOTHEN" in folder:
        marker="s"
        color="red"

    if "E" == folder[0]:
        marker = "x"
        color="k"
        markersize = 15


    
    if params["lbfgs_max_iterations"] == len(loss):
        
        marker = "d"
        color="k"
        ax1.plot([],[], linewidth=0, color="k", markersize=markersize, marker="d", label=folder + " (not converged)")


    ax1.loglog(params["alpha"], 
        # params["Jd_final"],
        abs(params["Jd_final"])/params["Jd_init"], 
        markersize=markersize, 
        marker=marker, color=color)

    ax2.semilogx(params["alpha"], 
        params["optimization_time_hours"],
        markersize=markersize, 
        marker=marker, color=color)

    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("J_init / J_final")

    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("Optimization time (hours)")


for ax in [ax1, ax2]:

    plt.sca(ax)

    # plt.axhline(params["Jd_init"])
    plt.plot([],[], linewidth=0, color="blue", marker="x", label="DG0")
    plt.plot([],[], linewidth=0, color="green", marker="o", label="DG1")
    plt.plot([],[], linewidth=0, color="red", marker="s", label="DG1, no smoothen")
    plt.legend()

ax2.set_title("Note! Different machines were used!", fontsize=16)
plt.close(fig2)

plt.show()
plt.close()