import os, pathlib, json
from parse import parse
import subprocess

import copy

matchstring = ""
# matchstring = "OCD"
# matchstring = "E100"



if "home/bastian" in os.getcwd():
    superfolder = "/home/bastian/D1/registration/"

else:
    superfolder = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/"

resultpaths = ["mriregistration_outputs",
"croppedmriregistration_outputs"
]

superfolder = pathlib.Path(superfolder)

interesting_hyperparameters = ["ocd", "smoothen", "alpha", "max_timesteps", "lbfgs_max_iterations"]

for resultpath in resultpaths:

    results = {} # set()

    bestlist = [1e16, 1e16, None]
    configs = [
        {"smoothen": True, "ocd": False, "best": copy.deepcopy(bestlist)},
        {"smoothen": False, "ocd": False, "best": copy.deepcopy(bestlist)},
        {"smoothen": True, "ocd": True, "best": copy.deepcopy(bestlist)},
    ]

    resultpath = superfolder / resultpath

    folders = os.listdir(resultpath)

    folders = [resultpath / x for x in folders if matchstring in x]


    for folder in sorted(folders):

        # if "OCD" in str(folder) or "NOSM" in str(folder):
        #     continue

        hyperparameters = None
        running = False
        failed = False

        try:
            hyperparameters = json.load(open(folder / "hyperparameters.json"))
        except FileNotFoundError:
            print(folder, "has no hyperparamters")
        
        command = "scontrol show jobid -dd " + str(hyperparameters["slurmid"])
        sretval = subprocess.run(command, shell=True, capture_output=True)#, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # print(hyperparameters["slurmid"], sretval.returncode)
        # continue
        if sretval.returncode == 0:
            running = True
        else:
            assert "Invalid job id specified" in str(sretval.stderr)
            # print("Invalid job id specified" in str(sretval.stderr))

        assert len(os.listdir(folder)) > 1
            # continue
        try:
            Jdinit = hyperparameters["Jd_init"]
            Jdfinal = hyperparameters["Jd_final"]
        except KeyError:

            logfile = [x for x in os.listdir(folder) if x.endswith("_log_python_srun.txt")]

            if len(logfile) == 0:
                pass
            elif len(logfile) > 1:
                raise ValueError
            else:
                logfile = logfile[0]

                file1 = open(folder / logfile, 'r')

                Lines = file1.readlines()

                Jdinit, Jdfinal = None, None

                losses = {}

                for line in reversed(Lines):
                    if "F = " in line:
                        number = line.replace(" ", "")
                        number = number.replace("F=", "")
                        Jdfinal = float(number)
                        # print(line, Jdfinal)
                        # break

                    if "At iterate " in line:
                        result = parse("Atiterate{}f={}|projg|={}", line.replace(" ", ""))
                        # print(result, line,  line.replace(" ", ""))
                        losses[result[0]] = float(result[1].replace("D", "e"))


                    if "At iterate    0" in line:
                        #     f=  1.64507D+03    |proj g|=  2.61434D-01
                        # print(line)
                        result = parse("At iterate    {}    f=  {}    |proj g|=  {}", line)


                        assert "D" in result[1]
                        Jdinit = float(result[1].replace("D", "e"))

                    if (Jdinit is not None) and (Jdfinal is not None):
                        break

                
                os.system("rm " + str(folder / "loss.txt"))
                for loss in sorted(list(losses.items())):
                    with open(folder / "loss.txt", "a") as myfile:
                        myfile.write(str(loss[1])+ ", ")
                

        if not running:

            if "optimization_time_hours" not in hyperparameters.keys() and ("OCD" not in str(folder)):
                failed = True
                # continue
            else:
                key = ""

                for name in interesting_hyperparameters:
                    if name == "alpha":
                        key += name + ":" + format(float(hyperparameters[name]), ".0e") + ","
                    elif name == "ocd":
                        if hyperparameters[name] == True:
                            key += "ocd"
                        else:
                            key += "   "
                    elif name == "smoothen":
                        if "OCD" in str(folder):
                            key += "           "
                        elif hyperparameters[name] == True:
                            key += "--smoothen "
                        else:
                            key += "           "

                    else:
                        key += name + ":" + str(hyperparameters[name]) + ","

                if "RS" in str(folder):
                    key += "RS"

                while len(key) < len("ocdRS           RSalpha:1e-05,RSmax_timesteps:1,RSlbfgs_max_iterations:100,RS") + 1:
                    key += " "

                assert hyperparameters["smoothen"] == (not hyperparameters["nosmoothen"])

                results[key] = int(Jdinit), int(Jdfinal), hyperparameters["slurmid"], folder

                for config in configs:
                    best = None
                    match = True
                    for key, item in config.items():
                        
                        if (key != "best") and (hyperparameters[key] != item):
                            
                            match = False

                    if match:
                        if config["best"][1] > Jdfinal:
                            config["best"] = [Jdinit, Jdfinal, folder]
                        

        print(folder.name)


            
        if (not running) and (not failed): # and "Finalstate.xdmf" in os.listdir(folder):
            print("done.")
            print("Jd=", format(Jdinit, ".2e"), "-->", format(Jdfinal, ".4e"),  "(", hyperparameters["lbfgs_max_iterations"], " LBFGS)")

        elif (not running) and failed:
            print("failed", hyperparameters["slurmid"])

        else:
            print("still running ?, running =", running)
            print(hyperparameters["slurmid"],)



        print()

    print("All results") #  (merging different LBFGS settings)")
    for key, r in sorted(results.items()):
        print(key, r[0], "-->", r[1])

    def print_best(x):
        print("** best for setting:", x[2].name, format(x[0], ".1e"), "-->", format(x[1], ".1e"))

        if "postprocessing" in os.listdir(x[2]) and len(os.listdir(x[2] / "postprocessing")) > 2:
            print("  -- has postprocessing")


    for config in configs:
        if None not in config["best"]:
            print_best(config["best"])

    ### PRINT SLURM IDS FOR JOB IF NECCESSARY

    # if "croppedmriregistration_outputs" in str(resultpath):
    #     print("All results") #  (merging different LBFGS settings)")
    #     for key, r in sorted(results.items()):
    #         print(key, "SLURMID", r[2], r[3])

    print("----------------------------------------------------------------------------------------------------------------------------------------------")