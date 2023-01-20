import os, pathlib, json, subprocess
from fenics import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from parse import parse
# from scipy.interpolate import CubicSpline
import argparse
from scipy.signal import savgol_filter
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

parser = argparse.ArgumentParser()
parser.add_argument("--resync", action="store_true", default=False)
parsersargs = vars(parser.parse_args())



def read_loss_from_log(file1, jd0, reg0=0):

    Lines = file1.readlines()


    loss, reg = [], []

    for line in Lines:

        if "Iter" in line:
            result = parse("Iter{}Jd={}Reg={}", line.replace(" ", ""))
            if (result is not None):
                lval = float(result[1])
                rval = float(result[2])
                # print(result[0], result[1])
                loss.append(lval)
                reg.append(rval)
        if "Assembled functional" in line:
            result = parse("Assembledfunctional,J={}", line.replace(" ", ""))
            if (result is not None):
                jd_0 = float(result[0])

    if jd0 is None:
        jd0 = jd_0
    
    assert jd0 is not None


    loss = [jd0] + loss 
    reg = [reg0] + reg


    loss = np.array(loss)
    reg = np.array(reg)

    if loss.size != reg.size:
        breakpoint()

    return loss, reg, jd0


dpi = None
figsize= None

subj1 = "abby"
res = 8
xmlfile1 = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/scripts/preprocessing/chp4/outs/abby/" + subj1 + str(res) + ".xml"
deformed_mesh = Mesh(xmlfile1)
quality = MeshQuality.radius_ratio_min_max(deformed_mesh)
meshes={}
meshes["input"] = {"min inner/outer radius" : quality[0], "Delta J = ": None}


# foldername = "croppedmriregistration_outputs"
# # sfoldername = "mriregistration_outputs"




fig1, ax1 = plt.subplots(dpi=dpi, figsize=figsize)
fig2, ax2 = plt.subplots(dpi=dpi, figsize=figsize)
fig3, ax3 = plt.subplots(dpi=dpi, figsize=figsize)

for foldername in ["croppedmriregistration_outputs2", "croppedmriregistration_outputs"]:

    localpath = pathlib.Path("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration") / foldername
    expath = pathlib.Path("/home/bastian/D1/registration") / foldername

    if foldername == "mriregistration_outputs":
        # runnames = [x for x in os.listdir(localpath) if x[0] == "E"]
        # jd0 = 2051.0663685198188
        """
        The coarsened resolution results
        """
        runnames = ['E100A0.0001LBFGS100NOSMOOTHEN',
                    'E100A0.01LBFGS100NOSMOOTHEN',
                    'E100A0.001LBFGS100',
                    'E100A0.0001LBFGS100',
                    'E100A0.01LBFGS500',
                    'E1000A0.01LBFGS500']

    elif foldername == "croppedmriregistration_outputs2":
        runnames = ["E100A1.0LBFGS100", "E100A0.01LBFGS100"]
    else:

        """
        The full resolution results
        """
        # jd0 = 18122.697155044116

        runnames = ['E100A0.01LBFGS100', 'E10A0.0001LBFGS100', 
        'E10A0.01LBFGS100', 'E1A0.01LBFGS100', 'E100A0.0001LBFGS100', 
        'E1A0.01LBFGS100NOSMOOTHEN', 'E100A0.01LBFGS100NOSMOOTHEN', 'E10A0.01LBFGS100NOSMOOTHEN', 
        'E10A0.0001LBFGS100NOSMOOTHEN', 'E1A0.0001LBFGS100']

    for runname in sorted(runnames):
        
        # breakpoint()
        # os.system("rm -r " + str(runname))
        
        if not (localpath / runname).is_dir():

            os.makedirs(localpath / runname, exist_ok=True)
        
        lossfile = localpath / runname / "loss.txt"
        hyperparameterfile = localpath / runname / "hyperparameters.json"

        if parsersargs["resync"]:

            command = "rsync -r "
            command += "ex:" + str(expath / runname / "*.txt")
            command += " "
            command += str(localpath / runname)

            subprocess.run(command, shell=True)

            

            command = "rsync -r "
            command += "ex:" + str(expath / runname / "*.json")
            command += " "
            command += str(localpath / runname)
            subprocess.run(command, shell=True)

        hyperparameters = json.load(open(hyperparameterfile))
        domain_size = np.product(hyperparameters["input.shape"])

        if "optimization_time_hours" in hyperparameters.keys():

            print(runname, "Compute time", hyperparameters["optimization_time_hours"])


        if "slurmid" in hyperparameters.keys() and parsersargs["resync"]:
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

        logfiles = [x for x in os.listdir(localpath / runname) if "_log_python_srun.txt" in x]
        assert len(logfiles) <= 1
        loss2, reg2 = None, None
        
        if len(logfiles) > 0:

            file1 = open(localpath / runname / logfiles[0], 'r')
            loss, reg, jd0 = read_loss_from_log(file1, jd0=None, reg0=0)

        startloss = None
        startjd0 = None

        if hyperparameters["starting_guess"] is not None:
            # load loss from starting guess run
            if not "/croppedmriregistration_outputs/" in hyperparameters["starting_guess"]:
                raise NotImplementedError
            
            startresultfolder = pathlib.Path("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration") / "croppedmriregistration_outputs" / runname
            startlogfiles = [x for x in os.listdir(startresultfolder) if "_log_python_srun.txt" in x]


            file1 = open(startresultfolder / startlogfiles[0], 'r')
            startloss, startreg, startjd0 = read_loss_from_log(file1, jd0=None, reg0=0)
            

            print(runname, "has starting guess")
        

        error = False
        cancelled = False

        if (localpath / runname / (str(hyperparameters["slurmid"]) + ".out")).is_file():
            outfile = open(localpath / runname / (str(hyperparameters["slurmid"]) + ".out"), "r")

            Lines = outfile.readlines()

            for line in Lines:
                if "valueerror" in line.lower():

                    error = True
                    print(line, "<------- error message in ", runname, "<----------")
                    break
                if "CANCELLED" in line:

                    cancelled = True
                    break

        try:
            loss2 = np.genfromtxt(lossfile, delimiter=",")# [:-1
            reg2 = np.genfromtxt(localpath / runname /"regularization.txt", delimiter=",")
            loss2[1:] = loss2[0:-1]
            loss2[0] = jd0
            reg2[1:] = reg2[0:-1]
            reg2[0] = 0
            if loss2.size == loss.size:
                loss = loss2
                reg = reg2
                print(runname, "Plotting loss from loss.txt instead of logfile")
        except FileNotFoundError:
            pass

        

        running = "Jd_final" not in hyperparameters.keys()
        try:
            if not np.allclose(hyperparameters["Jd_init"], jd0):
                print(runname)
                breakpoint()
                raise ValueError
        except KeyError:

            if error:
                print(runname, "error, ", hyperparameters["slurmid"])
            elif cancelled:
                print(runname, "cancelled, ", hyperparameters["slurmid"])
            elif running:
                print(runname, "not done", hyperparameters["slurmid"])

        linestlyle="-"
        if "OCD" in str(runname):
            raise ValueError

        # label = runname
        label = r"$\alpha$=" + format(hyperparameters["alpha"], ".0e") + "," + format(hyperparameters["max_timesteps"], ".0f") + " time steps"
        
        if error:
            label += "\n(error)"
        elif cancelled:
            label += "\n(cancelled)"
        elif running:
            label += "\n(running)"


        marker = None
        markevery= 1e14

        if "nosmoothen" in str(runname).lower():
            if "\n" not in label:
                label += "\n"
            label += "(no smoothen)"
            marker = "x"
            markevery= 10
            linestlyle = "--"



        loss /= domain_size

        reg /= domain_size

        if startjd0 is not None:
            jd0 = startjd0

        dJ = (loss + reg) / (jd0 / domain_size)
        iters = [1 + x for x in range(len(reg))]

        if startloss is not None:
            startloss /= domain_size
            startreg /= domain_size
            dJ = ((startloss + startreg) / (startjd0 / domain_size)).tolist() + dJ.tolist()
            iters = [1 + x for x in range(len(dJ))]

        p = ax1.plot(iters, dJ, linestyle=linestlyle, label=label, marker=marker, markevery=markevery)

        c = p[0].get_color()

        if len(dJ) > 10 and (not error) and (hyperparameters["alpha"] > 1e-4):
            label = label.replace("\n(cancelled)", "")
            
            if "(no smoothen)" in label and "\n" not in label:
                label = label.replace("(no smoothen)", "\n(no smoothen)")

            ax3.plot(iters, dJ , color=c, linestyle=linestlyle, label=label, marker=marker, markevery=markevery)

        ax2.semilogy([1 + x for x in range(len(reg))], reg, color=c, linestyle=linestlyle, label=label)



    # if "postprocessing" in os.listdir(localpath / runname):
    #     deformed_mesh = Mesh(str(localpath / runname / "postprocessing" / "transformed_input_mesh.xml"))
    #     quality = MeshQuality.radius_ratio_min_max(deformed_mesh)

    #     meshes[runname] = {"min inner/outer radius" : quality[0], "Delta J = ": ((loss + reg) / (jd0 / domain_size))[-1]}

    for ax in [ax1, ax2, ax3]:

        plt.sca(ax)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # plt.legend()
        plt.xlabel("LBFGS iteration")
        plt.ylabel("$L^2$-mismatch to target image")
        plt.tight_layout()



    ax2.set_ylabel("Regularization")
    ax1.set_ylabel(r"$\frac{J}{J(0)}$")
    ax3.set_ylabel(r"$\frac{J}{J(0)}$")
    # plt.savefig("./losses.png")

    # plt.close(fig2)
    # plt.close(fig1)

    for key, item in meshes.items():
        print(key, item)

plt.show()
