import os
import pathlib
import time
jobpath = "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/slurm/"

while True:
    for job in sorted(os.listdir(jobpath)):

        if not job.endswith(".out"):
            
            continue

        jobid = int(job.replace(".out", ""))

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

        for line in Lines:
            # print(line)

            if "error" in line.lower():
                # os.system("scancel " + jobid)
                # print(line)
                print("Cancelled job", jobid)
                break


    print("Sleeping for one hour")
    time.sleep(60 * 60)