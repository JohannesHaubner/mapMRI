import os
import pathlib
import time
jobpath = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/2dslurm/"

while True:
    for job in sorted(os.listdir(jobpath)):

        # if not job.endswith(".out"):
        #     continue

        if job.endswith(".out"):
            jobid = int(job.replace(".out", ""))
            if jobid < 429720:
                continue
        else:
            # continue
            jobid = job


        jobfile = jobpath + job

        jobid = str(pathlib.Path(job).stem)
        jobid = jobid.replace("_log_python_srun", "")

        file1 = open(jobfile, 'r')


        try:
            Lines = file1.readlines()
        except UnicodeDecodeError:
            print("UnicodeDecodeError at job", job, "will continue to next job")
            continue
        
        succes = False
        RK = False
        ELINE = False
        foldername = ""
        for line in Lines:

            if "timestepping : RungeKutta" in line:
                RK = True

            if "outfoldername : E" in line:
                # print(jobid)
                ELINE = True
                foldername = line

            if "error".lower() in line.lower():
                # print(line)
                succes = True

        # if succes:
        #     print(jobid)




        #    print(Lines[-5:])
    print("Sleeping for one hour")
    time.sleep(60 * 60)