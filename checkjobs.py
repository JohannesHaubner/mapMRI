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
            if jobid < 420480:
                continue
        else:
            jobid = job


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
            # eprint(line)

            if "    ".lower() in line.lower():
                succes = True

        if succes:
            print(jobid)

        #    print(Lines[-5:])
    print("Sleeping for one hour")
    time.sleep(60 * 60)