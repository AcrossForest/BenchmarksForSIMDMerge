#!/usr/bin/env python3

from simpleWorkflow import simpleWorkflowMain
from pathlib import Path
from supportedWorkloads import kernelList
baseName = Path(__file__).name.strip(".py")


m = 1000
edgeFactors = [10]
matrixList = [(m, m, m * e) for e in edgeFactors]

if __name__ == "__main__":
    simpleWorkflowMain(
        baseName=baseName,
        kernelList=kernelList,
        matrixList=matrixList
    )
