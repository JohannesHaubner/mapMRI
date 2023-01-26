from parse import parse
import numpy as np

nprocs = 16

def read_memory(filename):

    file = open(filename)
    Lines = file.readlines()

    mems = {}
    for line in Lines:

        if "outfoldername : " in line:
            outfoldername = parse("outfoldername : {}", line)[0]

        if not "Memory (TB) " in line:
            continue
        result=parse("Memory (TB) {} current_iteration {} process {}", line)
        try:
            mems[result[1]].append(float(result[0]))
        except KeyError:
            mems[result[1]] = [float(result[0])]

    mems2 = {}
    for key, item in mems.items():

        # if "cubeslurm" in str(filename):
        #     assert len(item) == nprocs

        mems2[key] = sum(item)


    mema = np.array(list(mems2.items())).astype(float)

    return mema, outfoldername

    
