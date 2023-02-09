import os
import pathlib
import time
jobpaths = [
# "/home/bastian/Oscar-Image-Registration-via-Transport-Equation/2dslurm/",
"/home/bastian/D1/registration/mrislurm/",
"/home/bastian/D1/registration/transportslurm/",
"/home/bastian/D1/registration/cubeslurm/",
]

while True:

    for jobpath in jobpaths:

        os.chdir(jobpath)

        for job in sorted(os.listdir(jobpath)):
            
            
            # if not job.endswith(".out"):
            #     continue

            if job.endswith(".out"):
                jobid = int(job.replace(".out", ""))
                if jobid < 439945:
                    continue
            else:
                continue

            jobfile = jobpath + job

            jobid = str(pathlib.Path(job).stem)
            file1 = open(jobfile, 'r')

            try:
                Lines = file1.readlines()
            except UnicodeDecodeError:
                print("UnicodeDecodeError at job", job, "will continue to next job")
                continue
            
            errormessage = False

            for line in Lines:


                if "error".lower() in line.lower():
                    errormessage = True
                    print(jobid)
                    break
            if errormessage:
                #if (os.system('scontrol show jobid -dd ' + str(jobid))) == 0:
                print("Killed", jobid)
                os.system("scancel " + str(jobid))




        #    print(Lines[-5:])
    print("Sleeping for one half hour")
    time.sleep(60 * 30)