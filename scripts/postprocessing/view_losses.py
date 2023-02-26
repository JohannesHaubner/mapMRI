import os, pathlib, json, subprocess
from fenics import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from parse import parse
import argparse
import nibabel

parser = argparse.ArgumentParser()
parser.add_argument("--resync", action="store_true", default=False)
parser.add_argument("--only", type=int, default=None)
parsersargs = vars(parser.parse_args())


def make_loss_history(hyperparameters, expath, expath2, localpath, loss, runname=None):

    init = False
    k = 0

    if hyperparameters["starting_state"]:
        key = "starting_state"
    elif hyperparameters["starting_guess"]:
        key = "starting_guess"

    current_start = hyperparameters[key]

    while not init and k < 32:

        
        starting_guess_dir = pathlib.Path(current_start.replace(str(expath.parent), str(localpath.parent))).parent
        if "global" in str(current_start):
            starting_guess_dir = pathlib.Path(current_start.replace(str(expath2.parent), str(localpath.parent))).parent

        try:
            assert starting_guess_dir.is_dir()
        except:
            breakpoint()
        starting_hyperparameters = json.load(open(starting_guess_dir / "hyperparameters.json"))
        # assert starting_hyperparameters["starting_state"] is None

        startlogfiles = [x for x in os.listdir(starting_guess_dir) if x.endswith("_log_python_srun.txt")]
        assert len(startlogfiles) == 1

        file2 = open(starting_guess_dir / startlogfiles[0], 'r')
        startloss, _ = read_loss_from_log(file2, hyperparameters=starting_hyperparameters)

        s0 = startloss[0, 1]




        x0 = np.zeros_like(startloss) + np.nan
        try:
            loss = np.vstack([x0, loss])
        except ValueError:
            breakpoint()
            exit()
            
        loss[:, 0] = list(range(0, loss.shape[0]))

        loss[:, 0] -= 1
        loss[:, 0] = np.abs(loss[:, 0])

        # exclude_keys = ["slurmid", "logfile", 
        #                 "readname", "starting_state", "lbfgs_max_iterations", "", "", 
        #                 "output_dir"]

        # for key, item in starting_hyperparameters.items():
            
        #     if key in exclude_keys:
        #         continue
            
        #     try:
        #         if not hyperparameters[key] == item:
        #             print(key, "is different:")
        #             print(starting_hyperparameters[key])
        #             print(hyperparameters[key])
        #     except KeyError:
        #         pass


        current_start = starting_hyperparameters[key]
        k += 1

        # print(k)

        # if k > 10:
        #     breakpoint()

        if current_start is None:
            init = True
            # breakpoint()
            return loss, s0



def read_loss_from_log(file1, hyperparameters):

    Lines = file1.readlines()

    start_values = [0, None, None, None]

    history = []

    line_searches = {}
    iterk = 1


    for idx, line in enumerate(Lines):

        if "Iter" in line:
            result = parse("Iter{}Jd={}L2loss={}", line.replace(" ", ""))
            if (result is not None):


                history.append([float(result[x]) for x in range(3)])

        # Assembled error between transported image and target, Jdata= 14579.269118718654
        # L2 error between transported image and target, Jdata_L2= 14579.269118718654

        if "Assembled error between transported image and target" in line:
            result = parse("Assembled error between transported image and target, Jdata= {}", line) # .replace(" ", ""))
            start_values[1] = float(result[0])
        if "L2 error between transported image and target, Jdata_L2" in line:
            result = parse("L2 error between transported image and target, Jdata_L2= {}", line) # .replace(" ", ""))
            start_values[2] = float(result[0])

        if line[:3] == "Reg":
            result = parse("Reg {}", line) # .replace(" ", ""))
            start_values[3] = float(result[0])

        if not iterk in line_searches.keys():
            if "At iterate    " + str(iterk) in line:
                for k in range(5):
                    if "LINE SEARCH" in Lines[idx + k]:
                        result = parse("LINESEARCH{}times;normofstep={}\n", Lines[idx + k].replace(" ", ""))
                        # breakpoint()
                        line_searches[iterk] = result[0]
                        iterk += 1
                #  LINE SEARCH           1  times; norm of step =    5.0000000000000000

        
        if None not in start_values and len(history) == 0:
            history.append(start_values)


    if len(history) > 0 and len(history[0]) == 4:
        history[0].pop(-1)

    history = np.array(history)

    if int(hyperparameters["slurmid"]) <= 444774: # or int(hyperparameters["slurmid"]) == 444925:
        # There was a bug with factor 0.5 during loss storage for huber loss

        print("*"*100)
        print("WARNING: Assuming job" + hyperparameters["slurmid"] + " is affected by the bug with factor 0.5 during storage of data loss!")
        print("*"*100)

        history[1:,1] *= 2
        

    return history, line_searches


dpi = None
figsize= None

subj1 = "abby"
res = 8
xmlfile1 = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/scripts/preprocessing/chp4/outs/abby/" + subj1 + str(res) + ".xml"
deformed_mesh = Mesh(xmlfile1)
quality = MeshQuality.radius_ratio_min_max(deformed_mesh)
meshes={}
meshes["input"] = {"min inner/outer radius" : quality[0], "Delta J = ": None}

foldernames = [ # "normalized-outputs",
                "ventricle-outputs"
                ]

if parsersargs["resync"]:

    for foldername in foldernames:
        localpath = pathlib.Path("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration") / foldername
        expath = pathlib.Path("/home/bastian/D1/registration") / foldername
        expath2 = pathlib.Path("/global/D1/homes/bastian/registration") / foldername

        os.makedirs(localpath, exist_ok=True)

        command = "ssh bastian@ex ls /home/bastian/D1/registration/" + foldername
        ret = subprocess.run(command, shell=True, capture_output=True)
        runnames = str(ret.stdout)[2:].split(r"\n")

        runnames_1 = [x for x in runnames if len(x) > 1]

        runnames = []
        for r in runnames_1:

            command = "ssh bastian@ex ls /home/bastian/D1/registration/" + foldername + "/" + r
            
            ret = subprocess.run(command, shell=True, capture_output=True)
            subfolders = str(ret.stdout)[2:].split(r"\n")

            if len(subfolders) == 2:
                runnames.append(r + "/" + subfolders[0])
            else:
                runnames.append(r)

        with open(localpath / "remote_folders.json", 'w') as outfile:
            json.dump({"folders": runnames}, outfile, sort_keys=True, indent=4)

        for runname in sorted(runnames):

            if parsersargs["only"] is not None and (str(parsersargs["only"]) not in runname):
                print("--only is set, continue")
                continue
            
            if not (localpath / runname).is_dir():

                os.makedirs(localpath / runname, exist_ok=True)
            
            lossfile = localpath / runname / "loss.txt"
            l2lossfile = localpath / runname / "l2loss.txt"
            hyperparameterfile = localpath / runname / "hyperparameters.json"

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

            command = "rsync -r "
            command += "ex:" + str(expath / runname / "CurrentState.mgz")
            command += " "
            command += str(localpath / runname)
            subprocess.run(command, shell=True)

            if "optimization_time_hours" in hyperparameters.keys():

                print(runname, "Compute time", hyperparameters["optimization_time_hours"])
                command = "rsync -r "
                command += "ex:" + str(expath / runname / "Finalstate.mgz")
                command += " "
                command += str(localpath / runname)
                subprocess.run(command, shell=True)

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



def slurmid(runname, localpath):
    hyperparameterfile = localpath / runname / "hyperparameters.json"
    hyperparameters = json.load(open(hyperparameterfile))
    return hyperparameters["slurmid"]


for foldername in foldernames:

    fig1, ax1 = plt.subplots(dpi=dpi, figsize=figsize)
    # fig2, ax2 = plt.subplots(dpi=dpi, figsize=figsize)
    fig3, ax3 = plt.subplots(dpi=dpi, figsize=figsize)



    localpath = pathlib.Path("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration") / foldername
    expath = pathlib.Path("/home/bastian/D1/registration") / foldername   
    expath2 = pathlib.Path("/global/D1/homes/bastian/registration") / foldername

    with open(localpath / "remote_folders.json", 'r') as outfile:
        runnames = json.load(outfile)["folders"]

    for runname in sorted(runnames, key=lambda x: slurmid(x, localpath)):

        lossfile = localpath / runname / "loss.txt"
        l2lossfile = localpath / runname / "l2loss.txt"
        hyperparameterfile = localpath / runname / "hyperparameters.json"
        hyperparameters = json.load(open(hyperparameterfile))
        domain_size = np.product(hyperparameters["target.shape"])

        inputfile = hyperparameters["input"].replace("/d1/", "/D1/").replace(str(expath.parent), str(localpath.parent))
        targetfile = hyperparameters["target"].replace("/d1/", "/D1/").replace(str(expath.parent), str(localpath.parent))

        if (localpath / runname / "Finalstate.mgz").is_file():
            print()
            print(hyperparameters["slurmid"])
            print()
            viewcommmand = "freeview "
            viewcommmand += inputfile + " "
            viewcommmand += targetfile + " "
            viewcommmand += str(localpath / runname / "Finalstate.mgz")
            print(viewcommmand)
            print()

        if (localpath / runname / "CurrentState.mgz").is_file():

            # assert nibabel.load(str(localpath / runname / "CurrentState.mgz")).get_fdata().shape == nibabel.load(targetfile).get_fdata().shape
            # assert np.allclose(nibabel.load(str(localpath / runname / "CurrentState.mgz")).affine, nibabel.load(targetfile).affine)
            # assert max(nibabel.load(str(localpath / runname / "CurrentState.mgz")).get_fdata().shape) < 200

            print()
            print(hyperparameters["slurmid"])
            print()
            viewcommmand = "freeview "
            viewcommmand += inputfile + " "
            viewcommmand += targetfile + " "
            viewcommmand += str(localpath / runname / "CurrentState.mgz")
            print(viewcommmand)
            print()

        if not hyperparameters["slurmid"] in ["445806", "445807"]:
            assert foldername in hyperparameters["output_dir"]

        logfiles = [x for x in os.listdir(localpath / runname) if x.endswith("_log_python_srun.txt")]
        # breakpoint()
        assert len(logfiles) == 1


        file1 = open(localpath / runname / logfiles[0], 'r')
        loss, line_searches = read_loss_from_log(file1, hyperparameters=hyperparameters)

        killed = False
        valueerror = False
        cancelled = False

        if (localpath / runname / (str(hyperparameters["slurmid"]) + ".out")).is_file():
            outfile = open(localpath / runname / (str(hyperparameters["slurmid"]) + ".out"), "r")

            Lines = outfile.readlines()

            for line in Lines:
                if "valueerror" in line.lower():
                    valueerror = True
                    print(line, "<------- error message in ", runname, "<----------")

                if "error" in line.lower():
                    killed = True
                    break

                if "CANCELLED" in line:
                    cancelled = True
                    break

        if (hyperparameters["starting_state"] is not None) or (hyperparameters["starting_guess"] is not None):

            loss2, startloss = make_loss_history(hyperparameters, expath,expath2, localpath, loss, runname=runname)

            startloss /= domain_size

            if loss2 is None:
                #breakpoint()
                pass
            else:
                loss = loss2

        else:
            try:
                startloss = loss[0, 1] / domain_size
            except:
                breakpoint()
                exit()

        if cancelled or killed or valueerror:
            # continue
            pass



        running = "Jd_final" not in hyperparameters.keys()


        if killed:
            print(runname, "killed, ", hyperparameters["slurmid"])
        elif valueerror:
            print(runname, "valueerror, ", hyperparameters["slurmid"])
        elif cancelled:
            print(runname, "cancelled, ", hyperparameters["slurmid"])
        elif running:
            print(runname, "not done", hyperparameters["slurmid"])

        if not running:
            print(runname, "done.")

        linestlyle="-"

        label = r"$\alpha$=" + format(hyperparameters["alpha"], ".0e") + "," + format(hyperparameters["max_timesteps"], ".0f") + " time steps"
        
        if killed:
            label += "\n(error)"
        elif cancelled:
            label += "\n(cancelled)"
        elif running:
            label += "\n(running)"

        label += hyperparameters["slurmid"]

        marker = None
        markevery= 1e14

        if hyperparameters["tukey"]:
            label += "(tukey, c=" + str(hyperparameters["tukey_c"]) + ")"
            linestlyle = "--"

        elif hyperparameters["huber"]:
            label += "(huber, c=" + str(hyperparameters["huber_delta"]) + ")"
            marker = "x"
        
        if hyperparameters["starting_guess"] is not None:
            label += "(restart)"
            # linestlyle = "--"

        loss[:, 1:] /= domain_size

        p = ax1.plot(loss[:, 0], loss[:, 2], linestyle=linestlyle, label=label, 
                    marker="o", #linewidth=0,
                    )        

        if len(loss[:,0]) > 1:
            ax3.plot(loss[:, 0], loss[:, 1] / startloss , color=p[0].get_color(), linestyle=linestlyle, label=label, 
                    # marker=marker, markevery=markevery
                )

    ax1.set_ylabel(r"$L^2$-loss   $\frac{1}{|\Omega|}\int_{\Omega}(\mathrm{State}-\mathrm{Target})^2\, dx$")
    ax3.set_ylabel(r"Reduction in Discrepancy Deformed-Target")


    print(hyperparameters["slurmid"], foldername, runname)

    for ax in [ax1, ax3]:
        plt.sca(ax)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel("LBFGS iteration")
        plt.tight_layout()



    plt.show()
