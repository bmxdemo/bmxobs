import sys
import bmxobs

dataName = sys.argv[1]

bins = (280, 300)

print("Setting up BMXObs {}".format(dataName))

D = bmxobs.BMXSingleFreqObs(dataName, freq_bins=bins)

print("{} Set Up".format(dataName))