import os
import json
import pathlib

p = pathlib.Path("/home/bastian/D1/registration/normalized-outputs")

for x,y,z in os.walk(p):
    if len(z) > 0:
        hf = pathlib.Path(x) / "hyperparameters.json"
        if hf.is_file():
            h = json.load(open(hf))
            print(h["starting_state"])