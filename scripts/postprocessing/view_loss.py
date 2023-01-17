import os, pathlib, json, subprocess

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from parse import parse
# FS = 40
# dpi = 400
# figsize = (10, 10)
# matplotlib.rcParams["lines.linewidth"] = 2
# matplotlib.rcParams["axes.linewidth"] = 2
# matplotlib.rcParams["axes.labelsize"] = FS # "xx-large"
# matplotlib.rcParams["grid.linewidth"] = 1
# matplotlib.rcParams["xtick.labelsize"] = FS # "xx-large"
# matplotlib.rcParams["ytick.labelsize"] = FS # "xx-large"
# matplotlib.rcParams["legend.fontsize"] = FS # "xx-large"
# matplotlib.rcParams["font.size"] = FS

def read_loss_from_log(file1):

    Lines = file1.readlines()


    loss , reg = [jd0], [0]

    for line in Lines:

        if "Iter" in line:
            result = parse("Iter{}Jd={}Reg={}", line.replace(" ", ""))
            if (result is not None):
                lval = float(result[1])
                rval = float(result[2])
                # print(result[0], result[1])
                loss.append(lval)
                reg.append(rval)


    loss = np.array(loss)
    reg = np.array(reg)

    if loss.size != reg.size:
        breakpoint()

    return loss, reg


dpi = None
figsize= None

foldername = "croppedmriregistration_outputs"
# foldername = "mriregistration_outputs"

localpath = pathlib.Path("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration") / foldername

expath = pathlib.Path("/home/bastian/D1/registration") / foldername



if foldername == "mriregistration_outputs":
    # runnames = [x for x in os.listdir(localpath) if x[0] == "E"]
    jd0 = 2051.0663685198188

    runnames = ['E100A0.0001LBFGS100NOSMOOTHEN',
 'E100A0.01LBFGS100NOSMOOTHEN',
 'E100A0.001LBFGS100',
 'E100A0.0001LBFGS100',
 'E100A0.01LBFGS500',
 'E1000A0.01LBFGS500']


else:
    jd0 = 18122.697155044116



    runnames = ['E100A0.01LBFGS100',
 'E10A0.0001LBFGS100',
 'E10A0.01LBFGS100',
 'E1A0.01LBFGS100',
 'E100A0.0001LBFGS100',
 'E1A0.01LBFGS100NOSMOOTHEN',
 'E100A0.01LBFGS100NOSMOOTHEN',
 'E10A0.01LBFGS100NOSMOOTHEN']


fig1, ax1 = plt.subplots(dpi=dpi, figsize=figsize)
fig2, ax2 = plt.subplots(dpi=dpi, figsize=figsize)
fig3, ax3 = plt.subplots(dpi=dpi, figsize=figsize)

for runname in sorted(runnames):
    
    os.system("rm -r " + str(localpath / runname))
    
    if not (localpath / runname).is_dir():

        os.makedirs(localpath / runname, exist_ok=True)
    
    lossfile = localpath / runname / "loss.txt"

    if not lossfile.is_file():

        command = "rsync -r "
        command += "ex:" + str(expath / runname / "*.txt")
        command += " "
        command += str(localpath / runname)

        subprocess.run(command, shell=True)

    hyperparameterfile = localpath / runname / "hyperparameters.json"

    # try:
    #     command = "rsync -r "
    #     command += "ex:" + str(expath / runname / "*_log_python_srun.txt")
    #     command += " "
    #     command += str(localpath / runname)
    #     subprocess.run(command, shell=True)
    # except:
    #     pass

    if not hyperparameterfile.is_file():
        command = "rsync -r "
        command += "ex:" + str(expath / runname / "*.json")
        command += " "
        command += str(localpath / runname)

        subprocess.run(command, shell=True)

    hyperparameters = json.load(open(hyperparameterfile))

    if hyperparameters["starting_guess"] is not None:

        print(runname, "has starting guess, ignoring for this plot")

        continue


    if "slurmid" in hyperparameters.keys() and not ("E100A0.01LBFGS100" == runname and foldername == "croppedmriregistration_outputs"):
        command = "rsync -r "
        command += "ex:" + "/home/bastian/D1/registration/mrislurm/" + str(hyperparameters["slurmid"]) + "_log_python_srun.txt"
        command += " "
        command += str(localpath / runname)
        retva = subprocess.run(command, shell=True, capture_output=True)

        command = "rsync -r "
        command += "ex:" + "/home/bastian/D1/registration/mrislurm/" + str(hyperparameters["slurmid"]) + ".out"
        command += " "
        command += str(localpath / runname)
        retva = subprocess.run(command, shell=True, capture_output=True)

        if retva.returncode == 0:
            command = "rsync -r "
            command += "ex:" + "/home/bastian/D1/registration/old/mrislurm/" + str(hyperparameters["slurmid"]) + "_log_python_srun.txt"
            command += " "
            command += str(localpath / runname)

            retva = subprocess.run(command, shell=True, capture_output=True)

            command = "rsync -r "
            command += "ex:" + "/home/bastian/D1/registration/mrislurm/" + str(hyperparameters["slurmid"]) + ".out"
            command += " "
            command += str(localpath / runname)
            retva = subprocess.run(command, shell=True, capture_output=True)
    
    assert foldername in hyperparameters["output_dir"]

    loss, reg = None, None

    if ("E100A0.01LBFGS100" == runname and foldername == "croppedmriregistration_outputs"):

        command = "rsync -r "
        command += "ex:" + "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/slurm/mrislurm/433513_log_python_srun.txt"
        command += " "
        command += str(localpath / runname)

        subprocess.run(command, shell=True)

        file1 = open(localpath / runname / "433513_log_python_srun.txt", 'r')

        loss, reg = read_loss_from_log(file1)

    # else:

    #     loss = np.genfromtxt(lossfile, delimiter=",")# [:-1]
    #     reg = np.genfromtxt(localpath / runname /"regularization.txt", delimiter=",")
    #     loss[1:] = loss[0:-1]
    #     loss[0] = jd0

    logfiles = [x for x in os.listdir(localpath / runname) if "_log_python_srun.txt" in x]

    loss2, reg2 = None, None

    if len(logfiles) > 0:

        file1 = open(localpath / runname / logfiles[0], 'r')
        loss, reg = read_loss_from_log(file1)
    

    error = False
    if (localpath / runname / (str(hyperparameters["slurmid"]) + ".out")).is_file():
        outfile = open(localpath / runname / (str(hyperparameters["slurmid"]) + ".out"), "r")

        Lines = outfile.readlines()

        for line in Lines:
            if "error" in line.lower():

                error = True
                print(line, "<------- error message in ", runname, "<----------")
                break

    try:
        loss2 = np.genfromtxt(lossfile, delimiter=",")# [:-1
        reg2 = np.genfromtxt(localpath / runname /"regularization.txt", delimiter=",")
        loss2[1:] = loss2[0:-1]
        loss2[0] = jd0

        if loss2.size == hyperparameters["lbfgs_max_iterations"]:
            loss, reg = loss2, reg2
            breakpoint()
            # TODO FIXME 
            loss2, reg2 = None, None

    except:
        pass

    
    if len(logfiles) > 1:

        breakpoint()
        file1 = open(localpath / runname / logfiles[1], 'r')

        loss2, reg2 = read_loss_from_log(file1)

    domain_size = np.product(hyperparameters["input.shape"])

    try:
        if not np.allclose(hyperparameters["Jd_init"], jd0):
            print(runname)
            breakpoint()
            raise ValueError
    except KeyError:
        print(runname, "not done")


    linestlyle="-"
    if "OCD" in runname:
        linestlyle = "--"

    # label = runname
    label = r"$\alpha$=" + format(hyperparameters["alpha"], ".0e") + "," + format(hyperparameters["max_timesteps"], ".0f") + " time steps"
    
    if error:
        label += "error!"

    marker = None
    markevery= 1e14

    if "nosmoothen" in runname.lower():
        label += "(no smoothen)"
        marker = "x"
        markevery= 10


    if loss is not None:

        loss /= domain_size

        reg /= domain_size

        p = ax1.plot([1 + x for x in range(len(loss))], loss, linestyle=linestlyle, label=label, marker=marker, markevery=markevery)

        c = p[0].get_color()

        ax2.semilogy([1 + x for x in range(len(reg))], reg, color=c, linestyle=linestlyle, label=label)

        for lossx in [loss2]:
            if lossx is None:
                continue
            lossx /= domain_size

            ax1.plot([1 + x for x in range(len(lossx))], lossx, color=c, linestyle=":", label=label + "from loss.txt", marker=marker, markevery=markevery)

        ax3.semilogy([1 + x for x in range(len(reg))], loss + reg, color=c, linestyle=linestlyle, label=label)


for ax in [ax1, ax2, ax3]:

    plt.sca(ax)

    plt.legend()
    plt.xlabel("LBFGS iteration")
    plt.ylabel("$L^2$-mismatch to target image")
    plt.tight_layout()



ax2.set_ylabel("Regularization")
ax3.set_ylabel("$J = J_d + J_reg$")
# plt.savefig("./losses.png")

plt.close(fig2)
plt.close(fig1)

plt.show()
