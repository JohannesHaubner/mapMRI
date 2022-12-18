import os
import pathlib
import pandas
import numpy as np

import json

alphas = ["1e-2", "1e-4", "1e-6", "1e-8"]
maxiters = [64, 128, 256, 512]

smoothenings = [True, False]

outputdir = pathlib.Path("/home/bastian/D1/hyperparameterstudy_2d/")


rows = []

for alpha in alphas:
    for maxiter in maxiters:
        for smoothen in smoothenings:
            subfoldername = "N" + str(maxiter) + "A" + alpha

            if smoothen:
                subfoldername = "smooth" + subfoldername

            subfolder = outputdir / subfoldername

            if not (subfolder / "hyperparameters.json").is_file():
                continue
            
            if not (subfolder / "_finalsolution.png").is_file():
                continue

            try:
                hyperparameters = json.load(open(subfolder / "hyperparameters.json"))

            except json.decoder.JSONDecodeError:
                print("vim " + str(subfolder / "hyperparameters.json"))

            if "krylov_failed" in hyperparameters.keys():
                breakpoint()

            
            loss_history = np.genfromtxt(subfolder / "loss.txt", delimiter=',')[:-1]
            
            header = ["alpha", "maxiter", "smoothen", "red. in Jd (%)", "epochs"]

            rows.append([alpha, maxiter, smoothen, format(100 * hyperparameters["Jd_final"] / hyperparameters["Jd_init"], ".0f"), loss_history.size])
            
            print(alpha, maxiter, smoothen, # format(hyperparameters["Jd_init"], ".2e"), 
                        format(hyperparameters["Jd_final"], ".6e"), loss_history.size, 
                        # hyperparameters["optimization_time_hours"], 
                        # hyperparameters["slurmid"]
                        )


df = pandas.DataFrame(rows, columns=header)

print(df)