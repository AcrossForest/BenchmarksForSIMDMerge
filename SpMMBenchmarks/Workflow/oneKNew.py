#!/usr/bin/env python3

from twoMatWorkflow import twoMatWorkflowMain
from pathlib import Path
from supportedWorkloads import kernelList
from workflow import MatrixSetting
baseName = Path(__file__).name.strip(".py")


m = 20
k = 10000
n = 10000

edgeFactors = range(10, 30, 10)
matrixList = [(MatrixSetting(m,k,e),
                MatrixSetting(k,n,e)) for e in edgeFactors]

if __name__ == "__main__":
    twoMatWorkflowMain(
        baseName=baseName,
        kernelList=kernelList,
        matrixList=matrixList
    )
