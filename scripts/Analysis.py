#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt

vigorColor = [255.0/255.0, 95.0/255.0, 31.0/255.0]
spacingColor = [33.0/255.0, 252.0/255.0, 13.0/255.0]
canopyColor = [0.0/255.0, 240.0/255.0, 255.0/255.0]


def extractData(pidDir, fname):
    file = f"../user_data/{pidDir}/{fname}"
    try:
        print(f"Opening the file {file}")
        with open(file, 'r') as fp:
            data = json.load(fp)
            return data
    except FileNotFoundError:
        print(f"Could not find {fname} at {pidDir}")



def plotTreeData(data):
    
    fig, axs = plt.subplots(1, 1)
    totalTime = data["Pruning Time"]

    # Plotting three lines 
    #   1) Between Times (y=1)
    #   2) Cut Time (y = 2)
    #   3) Decision Time (y = 3)

    cutData = data["Cut Data"][-1] # GETTING THE LAST COMPLETE DATA POINTS FOR REVIEW

    betweenTimes = cutData["Between Time"]
    cutTimes = cutData["Cut Time"]
    ruleTimes = cutData["Rule Time"]
    cuts = cutData["Rule"]

    vigorCutsTime = []
    budSpacingCutsTime = []
    canopyCutsTime = []

    xData = []
    yData = []

    startTime = 0
    # Loop Through the User Data
    for i in range(len(cutTimes)):
        # Between Times
        xData.append(startTime)
        yData.append(1)
        startTime += betweenTimes[i]
        xData.append(startTime)
        yData.append(1)

        # Cut Times
        xData.append(startTime)
        yData.append(2)
        startTime += cutTimes[i]
        xData.append(startTime)
        yData.append(2)

        # Rule Times
        xData.append(startTime)
        yData.append(3)
        startTime += ruleTimes[i]
        xData.append(startTime)
        yData.append(3)

        # Add vertical lines
        if cuts[i] == "Vigor":
            vigorCutsTime.append(startTime)
        elif cuts[i] == "Spacing":
            budSpacingCutsTime.append(startTime)
        else:
            canopyCutsTime.append(startTime)
    
    axs.plot(xData, yData, color="black", linestyle=":", label="Timing")
    axs.set_xlim(0, totalTime)
    labels = {"Vigor": "Branch Vigor", "Spacing": "Bud Spacing", "Canopy": "Canopy Cuts"}

    # Plot the vertical lines with the cuts
    for v in vigorCutsTime:
        axs.axvline(v, color=vigorColor, label=labels["Vigor"])
        labels["Vigor"] = "_nolegend_"
    for b in budSpacingCutsTime:
        axs.axvline(b, color=spacingColor, label=labels["Spacing"])
        labels["Spacing"] = "_nolegend_"
    for c in canopyCutsTime:
        axs.axvline(c, color=canopyColor, label=labels["Canopy"])
        labels["Canopy"] = "_nolegend_"

    axs.set_yticks([1, 2, 3], ["Between Cuts", "Cut Times", "Rule Times"])
    axs.set_xlabel("Total Pruning Time (s)")
    plt.legend()

    plt.show()



if __name__ == "__main__":
    pidDir = input("What folder in user_data are you analyzing?\n")
    fname = input("What json file are you analyzing?\n")

    data = extractData(pidDir, fname)
    if fname == "Evaluations.json" and data is not None:
        pass
    elif data is not None:
        plotTreeData(data)
    

