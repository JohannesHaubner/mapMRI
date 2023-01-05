import os
import json

path = "/home/bastian/D1/registration/timing"

os.chdir(path)

for f in os.listdir(path):

    

    par = json.load(open(f + "/hyperparameters.json"))

    print(f, par["optimization_time_hours"], par["Jd_init"], par["Jd_final"])
