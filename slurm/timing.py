import os, json, pathlib
import pandas

df = {}


for pp in ["/home/bastian/D1/registration/coarsedtiming", "/home/bastian/D1/registration/croppedtiming"]:

    p = pathlib.Path(pp)


    for f in os.listdir(p):

        f = p / f

        print(f)

        try:
            h = json.load(open(f / "hyperparameters.json"))
            df[f.name] = [ h["max_timesteps"], format(sum(h["times"])/len(h["times"]), ".2f"), h["processes"], "cropped" in str(pp)]

        except (FileNotFoundError, KeyError) as e:
            pass

header = ["timesteps", "time per call", "processes", "Full resolution"]


df = pandas.DataFrame.from_dict(df, orient="index", columns=header)

df.sort_values(["Full resolution", "processes", "timesteps",], inplace=True)

print(df)