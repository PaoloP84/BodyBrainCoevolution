#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sys
import os



found = False
averagen = 0
data = []
if len(sys.argv) == 1:
    cpath = os.getcwd()
    files = os.listdir(cpath)
    for f in files:
        if ("S" in f) and (".fit" in f):
            f = open(f)
            for l in f:
                for el in l.split():
                    if found:
                        averagen += 1
                        data.append(float(el))
                        found = False
                    if (el == 'bestgfit'):
                        found = True
print("")
if (averagen > 0):
    print("Average Generalization: %.2f +-%.2f (%d S*.fit files)" % (np.average(data), np.std(data), averagen))
else:
    print("No data found")
    print("Compute the average and stdev of generalization performance")
    print("Extract data from S*.fit files: data should follow the bestgfit key")



