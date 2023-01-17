import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("num")

ap = vars(ap.parse_args())

print("Starting!")
print(ap)
print("Sleeping 1 min")
time.sleep(60)
print("done")