#!/usr/bin/env python3

from simpleWorkflow import simpleWorkflowMain
from pathlib import Path
from supportedWorkloads import kernelList
baseName = Path(__file__).name.strip(".py")



matrixList = [(10,10,10 * n) for n in [2,3,4]]

if __name__ == "__main__":
    simpleWorkflowMain(
        baseName=baseName,
        kernelList=kernelList,
        matrixList=matrixList
    )

            
                

        


