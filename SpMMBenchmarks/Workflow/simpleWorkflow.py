from workflow import *
import sys
from typing import List, Dict, Tuple



def simpleWorkflowMain(baseName: str,
                       kernelList: Dict[str, str],
                       matrixList: List[Tuple[int, int, int]]):
    didAnything = False
    if "makeMat" in sys.argv:
        didAnything = True
        generateGroup(matrixList, globalDataDir/baseName, baseName)

    if "makeWorkload" in sys.argv:
        didAnything = True
        matList = loadMatrixDescList(
            globalDataDir/baseName/matrixMetaFile(baseName))
        for shortName, fullName in kernelList.items():
            generateSPGemmWorkload(
                abList=[(m, m, fullName) for m in matList],
                metafileDir=globalWorkloadDir/baseName/shortName,
                baseName=baseName
            )

    if "exec" in sys.argv:
        didAnything = True
        for shortName, fullName in kernelList.items():
            if (not shortName in sys.argv and
                    not fullName in sys.argv and
                    not "all" in sys.argv):
                print(f"Skip {shortName}:{fullName}\n")
                continue
            runAllWorkloadInMeta(
                globalWorkloadDir/baseName/shortName/workloadMetaFile(baseName)
            )
    if "execGem5" in sys.argv:
        didAnything = True
        for shortName, fullName in kernelList.items():
            if (not shortName in sys.argv and
                    not fullName in sys.argv and
                    not "all" in sys.argv):
                print(f"Skip {shortName}:{fullName}\n")
                continue
            runAllWorkloadInMeta(
                globalWorkloadDir/baseName/shortName/workloadMetaFile(baseName),
                useGem5=True
            )
    if "execGem5Parallel" in sys.argv:
        didAnything = True
        allThreads = [] # type: List[Thread]
        for shortName, fullName in kernelList.items():
            if (not shortName in sys.argv and
                    not fullName in sys.argv and
                    not "all" in sys.argv):
                print(f"Skip {shortName}:{fullName}\n")
                continue
            newThreads = runAllWorkloadInMeta(
                globalWorkloadDir/baseName/shortName/workloadMetaFile(baseName),
                useGem5=True,parallel=True
            )
            allThreads.extend(newThreads)
        for i,t in enumerate(allThreads):
            print(f"Finished one. {i}/{len(allThreads)}")
            t.join()
        print("All thread finished.")
    if "report" in sys.argv:
        didAnything = True
        allReportPaths = []
        for shortName, fullName in kernelList.items():
            if (not shortName in sys.argv and
                    not fullName in sys.argv and
                    not "all" in sys.argv):
                print(f"Skip {shortName}:{fullName}\n")
                continue
            allReportPaths.append(
                Path(globalWorkloadDir/baseName /
                     shortName/workloadMetaFile(baseName))
            )
        if len(allReportPaths) == 0:
            print("Do nothing: No kernel selected for reporting.")

        gatherReports(
            allReportPaths=allReportPaths,
            finalReportPath=globalcsvReportDir/defaultJsonFileName(baseName),
            finalReportCSVPath=globalcsvReportDir/defaultCSVFileName(baseName),
        )

    if not didAnything:
        print("Did no thing.\n")
