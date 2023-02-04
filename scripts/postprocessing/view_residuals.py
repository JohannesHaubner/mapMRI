import os, pathlib, json, subprocess
from fenics import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from parse import parse
# from scipy.interpolate import CubicSpline
import argparse
from scipy.stats import norm

foldername = "affine-rotated-outputs_noscale"

localpath = pathlib.Path("/home/basti/programming/Oscar-Image-Registration-via-Transport-Equation/registration") / foldername
expath = pathlib.Path("/home/bastian/D1/registration/") / foldername

subfolders = ["E70A0.01LBFGS20C1/", 
                #"tukey/E100A0.01LBFGS100/", "l2/E100A0.01LBFGS100/"
            ]


# subprocess.run("rsync -r ex:" + "/home/bastian/D1/registration/mrislurm/436253_log_python_srun.txt" + " " + str(localpath / "tukey/E100A0.01LBFGS100/" / ""), shell=True)
# subprocess.run("rsync -r ex:" + "/home/bastian/D1/registration/mrislurm/436252_log_python_srun.txt" + " " + str(localpath / "l2/E100A0.01LBFGS100/" / ""), shell=True)

for subfolder in subfolders:

    os.makedirs(localpath / subfolder, exist_ok=True)

    # print(str(expath /subfolder / ""))
    # exit()

    for file in [#"*.npy", 
                    "*.txt", "*.json"]:
        
        subprocess.run("rsync -r ex:" + str(expath /subfolder / file) + " " + str(localpath /subfolder / ""), shell=True)

    res = []
    nres = []
    for rfile in [x for x in os.listdir(localpath / subfolder) if ".npy" in x]:
        if "normalized" in rfile:
            nres.append(np.load(localpath / subfolder / rfile))
        else:
            res.append(np.load(localpath / subfolder / rfile))

    std = None

    if "tukey" in subfolder:

        file1 = open(str(localpath / "tukey/E100A0.01LBFGS100/" / "436253_log_python_srun.txt"))

        Lines = file1.readlines()

        
        for line in Lines:
            if "std_residual" in line and "same" not in line:
                result = parse("std_residual{}", line.replace(" ", ""))
                # assert len(result) == 1
                std = float(result[0])

    res = np.concatenate(res)
    
    bins = 100
    density = True

    plt.figure()
    plt.title(subfolder)
    plt.hist(res, bins=bins, density=density, label="residual")
    if std is not None:

        x = np.linspace(min(res), max(res), 100)
        plt.plot(x, norm.pdf(x, scale=std), color="green", label=r"Normal distribution $f(x, \sigma)$")
        plt.axvline(-std, ymax=norm.pdf(-std, scale=std),color="red")
        plt.axvline(std, ymax=norm.pdf(std, scale=std),color="red", label=r"$\pm \sigma$")

        plt.axvline(-std*4, color="k")
        plt.axvline(std*4, color="k", label=r"$\pm 4 \sigma$")

    plt.legend()
    # plt.yscale("log")
    # plt.show()
    # exit()

    if len(nres) > 0:
        nres = np.concatenate(nres)
        plt.figure()
        plt.title(subfolder + "(normalized)")
        plt.hist(nres, bins=bins, density=density)

        plt.axvline(-4, color="k")
        plt.axvline(4, color="k", label=r"$\pm 4 \sigma$")

        plt.yscale("log")

        print(nres[np.abs(nres) > 1].size, "/", nres.size, "points are outliers")
        print(nres[np.abs(nres) > 1].size / nres.size, "outliers")

plt.show()