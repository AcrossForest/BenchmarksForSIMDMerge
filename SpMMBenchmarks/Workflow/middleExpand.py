#!/usr/bin/env python3

from twoMatWorkflow import twoMatWorkflowMain
from pathlib import Path
from supportedWorkloads import kernelList
from workflow import MatrixSetting
baseName = Path(__file__).name.strip(".py")


m = 20
k = 3600
uniqueCols = 0
n = 1000 * 1000
# n = uniqueCols * 1000

edgeFactors = [30,60,90,120]
matrixList = [(MatrixSetting(m,k,e),
                MatrixSetting(k,n,e,uniqueCols)) for e in edgeFactors]

if __name__ == "__main__":
    twoMatWorkflowMain(
        baseName=baseName,
        kernelList=kernelList,
        matrixList=matrixList
    )
