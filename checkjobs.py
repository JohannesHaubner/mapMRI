import os
import pathlib
import time
jobpath = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/debug/"

while True:
    for job in sorted(os.listdir(jobpath)):

        if not job.endswith(".out"):
            
            continue
        # jobid = int(job.replace("_log_python_srun.txt", ""))
        jobid = int(job.replace(".out", ""))
        # print(jobid)
        if jobid < 420480:
            continue

        jobfile = jobpath + job

        jobid = str(pathlib.Path(job).stem)

        file1 = open(jobfile, 'r')


        try:
            Lines = file1.readlines()
        except UnicodeDecodeError:
            print("UnicodeDecodeError at job", job, "will continue to next job")
            continue
        succes = False
        for line in Lines:
            # print(line)

            if "success".lower() in line.lower():
                succes = True

        if not succes:
            print(jobid)
            print(Lines[-5:])
    print("Sleeping for one hour")
    time.sleep(60 * 60)