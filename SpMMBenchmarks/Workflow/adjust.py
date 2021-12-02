#!/usr/bin/env python3

from simpleWorkflow import simpleWorkflowMain
from pathlib import Path
from supportedWorkloads import kernelList
baseName = Path(__file__).name.strip(".py")



matrixList = [(100,100,100 * n) for n in [10]]

if __name__ == "__main__":
    simpleWorkflowMain(
        baseName=baseName,
        kernelList=kernelList,
        matrixList=matrixList
    )

            
                

        


