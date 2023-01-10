import os, pathlib, json
from parse import parse

matchstring = ""
# matchstring = "OCD"
# matchstring = "E100"

if "home/bastian" in os.getcwd():
    resultpath = "/home/bastian/D1/registration/mriregistration_outputs/"

else:
    resultpath = "/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration/mriregistration_outputs/"

resultpath = pathlib.Path(resultpath)

folders = os.listdir(resultpath)

folders = [resultpath / x for x in folders if matchstring in x]

for folder in sorted(folders):

    try:
        hyperparameters = json.load(open(folder / "hyperparameters.json"))
    except FileNotFoundError:
        print(folder, "has no hyperparamters")
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
            
    print(folder.name, "Jd=", format(Jdinit, ".2e"), "-->", format(Jdfinal, ".2e"))

    

    for root, dirs, files in os.walk(folder):
        
        # print(root)
        # print(dirs)
        # print(files)

        break
    
    if "Finalstate.xdmf" in os.listdir(folder):
        print(folder.name, "converged/done. (", hyperparameters["lbfgs_max_iterations"], " LBFGS)")

    else:
        print(folder.name, "still running ?")



    print()