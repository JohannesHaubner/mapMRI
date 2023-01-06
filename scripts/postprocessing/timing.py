import os
import json

paths = {"2d:": "/home/bastian/D1/registration/timing", "3d:": "/home/bastian/D1/registration/timing3d",}


for dim, path in paths.items():
    os.chdir(path)
    
    print("-"*80)
    print("Dimension:", dim)

    for f in sorted(os.listdir(path), key=lambda x: int(x[19:])):


        cores = f[19:]

        par = json.load(open(f + "/hyperparameters.json"))

        if "optimization_time_hours" not in par.keys():
            print(f, "not done")
            continue

        print(f, "cores=", cores, "compute time:", par["optimization_time_hours"]) # , par["Jd_init"], par["Jd_final"])
