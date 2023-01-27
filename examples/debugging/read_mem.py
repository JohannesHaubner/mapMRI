from parse import parse
import numpy as np

nprocs = 16

def read_memory(filename):

    outfoldername = ""

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


def check(filename):

    assert "out" not in str(filename)

    file = open(filename)
    Lines = file.readlines()

    for line in Lines:

        if "Transforming l2 control to L2 control" in line:
            assert "none" not in str(filename)

        if "Created Krylov solver in Preconditioning()" in line:
            assert "none" not in str(filename)
            assert "NOSMOOTHEN" not in str(filename)

        if "Setting velocity = l2_controlfun" in line:
            found_1 = True

    
    if "none" in str(filename):
        assert found_1



def check_for_error(filename):

    assert "out" in str(filename)

    file = open(filename)
    Lines = file.readlines()

    for line in Lines:


        if "error" in line:
            print(filename)
            print("--", line)
            return True

        if "success" in line:
            return True
    
