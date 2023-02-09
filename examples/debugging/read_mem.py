from parse import parse
import numpy as np

nprocs = 16

def read(filename):


    ntasks, maxcor, meshn = None, None, 32

    file = open(filename)
    Lines = file.readlines()

    for idx, line in enumerate(Lines):

        if "NTASKS" in line:
            ntasks = parse("NTASKS={}", line)[0]
        if "NCOR" in line:
            maxcor = parse("NCOR={}", line)[0]
        if "meshn" in line:
            meshn = parse("meshn={}", line)[0]
    

    
    return ntasks, maxcor, meshn



def read_memory(filename):

    line_searches = {}
    iterk = 1


    outfoldername = ""

    file = open(filename)
    Lines = file.readlines()

    mems = {}
    for idx, line in enumerate(Lines):

        if "outfoldername : " in line:
            outfoldername = parse("outfoldername : {}", line)[0]
        if "outputfolder : " in line:
            outputfolder = parse("outputfolder : {}", line)[0].replace("\n", "")


        if "Memory (TB) " in line:
            result=parse("Memory (TB) {} current_iteration {} process {}", line)
            try:
                mems[result[1]].append(float(result[0]))
            except KeyError:
                mems[result[1]] = [float(result[0])]


        if not iterk in line_searches.keys():
            if "At iterate    " + str(iterk) in line:
                for k in range(9):
                    if "LINE SEARCH" in Lines[idx + k]:
                        result = parse("LINESEARCH{}times;normofstep={}\n", Lines[idx + k].replace(" ", ""))
                        # breakpoint()
                        line_searches[iterk] = int(result[0])
                        iterk += 1

                        break

                # print(result)
                # exit()

    mems2 = {}
    kk = 0
    for key, item in mems.items():

        # if "cubeslurm" in str(filename):
        #     assert len(item) == nprocs

        if kk > 0 and len(mems[list(mems.keys())[0]]) != len(item):

            print("*"*80, "WARNING: last iterate did not print memory of all processes")

            return np.array(list(mems2.items())).astype(float), outfoldername, line_searches, outputfolder
        else:
            mems2[key] = sum(item)

        # assert len(mems[list(mems.keys())[0]]) == len(item)
        
        kk += 1
    mema = np.array(list(mems2.items())).astype(float)

    # print("--", np.max(mema[:, 1]))

    return mema, outfoldername, line_searches, outputfolder


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
    
