import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from systemCommon import *
from threading import Thread
import pandas as pd

################################################################
#              The default path and file names
###############################################################

# randomGeneratorPath = builtDir/"RandomGenerator"/exeName("randomGenerator")
randomGeneratorPath = Path(__file__).parent.parent / "randomGenerator"
harnessPath = builtDir/"Harness"/exeName("Harness")



globalDataDir = projectRoot.joinpath("data")
globalWorkloadDir = projectRoot.joinpath("workload")
globalcsvReportDir = projectRoot.joinpath("csvResults")
# globalDataDir = Path("../data/")
# globalWorkloadDir = Path("../workload")
# globalcsvReportDir = Path("../csvResults")


def matrixMetaFile(baseName:str) -> str:
    return f"{baseName}-metainfo.json"

def workloadMetaFile(baseName:str) -> str:
    return f"{baseName}-workload-metainfo.json"

def getReportFilePath(workloadDescFilePath:Path):
    return workloadDescFilePath.with_name(workloadDescFilePath.stem+"-report.json")

def defaultCSVFileName(baseName:str) -> str:
    return f"{baseName}-csvReport.csv"

def defaultJsonFileName(baseName:str) -> str:
    return f"{baseName}-jsonReport.json"


################################################################
#              Data structures
###############################################################


class sparseMatrixDesc:
    def __init__(self,fileName,m=None,n=None,nnz=None):
        self.fileName = fileName
        self.m = m
        self.n = n
        self.nnz = nnz
        self.edgeFactor = None
        if m is not None and nnz is not None:
            self.edgeFactor = nnz/m
    def toDict(self):
        # Apparentaly, this is simple
        return self.__dict__

    @staticmethod
    def fromDict(jsondict):
        obj = sparseMatrixDesc(
            fileName=jsondict["fileName"],
            m = jsondict["m"],
            n = jsondict["n"],
            nnz = jsondict["nnz"],
        )
        if abs(obj.edgeFactor - jsondict["edgeFactor"])>1e-3:
            raise Exception(f"Unreasonable edge factor in jsondict = {jsondict}")
        return obj

class workloadDesc:
    def __init__(self,kernelName:str,inputs:Dict[str,sparseMatrixDesc],outputs:Dict[str,sparseMatrixDesc]):
        self.kernelName = kernelName
        self.inputs = inputs # type: Dict[str,sparseMatrixDesc]
        self.outputs = outputs # type: Dict[str,sparseMatrixDesc]
        # self.outputs : List[sparseMatrixDesc] = []
    
    def toDict(self):
        return {
            "kernelName":self.kernelName,
            "inputs":{name:desc.toDict() for name,desc in self.inputs.items()},
            "outputs":{name:desc.toDict() for name,desc in self.outputs.items()},
        }



class reportFile:
    def __init__(self,jsondict):
        self.kernelName = jsondict["kernelName"] # type: str
        self.inputs = {name:sparseMatrixDesc.fromDict(d) for name,d in jsondict["inputs"].items()} # type: Dict[str,sparseMatrixDesc]
        self.outputs = {name:sparseMatrixDesc.fromDict(d) for name,d in jsondict["outputs"].items()} # type: Dict[str,sparseMatrixDesc]
        self.timmer = jsondict["timmer"]



################################################################
#               Generate random matrix
###############################################################
from dataclasses import dataclass
@dataclass
class MatrixSetting:
    m : int
    n : int
    nnzRow : int
    uniqueCols : int = 0

def generateMatNew(ms:MatrixSetting,metafileDir:Path,baseName:str)-> sparseMatrixDesc:
    if not randomGeneratorPath.exists():
        raise RuntimeError(f"Can't find random generator at {randomGeneratorPath}!")
    fileName = metafileDir.resolve()/f"{baseName}-{ms.m}-{ms.n}-{ms.nnzRow}-{ms.uniqueCols}.bincsr"
    print(f"Generating {fileName.__str__()}")
    nnz = ms.m * ms.nnzRow
    subprocess.run(
        args = [randomGeneratorPath.__str__(),str(ms.m),str(ms.n),str(nnz),str(ms.uniqueCols),fileName.with_suffix(".temp").__str__()],check=True)
    subprocess.run(
        args = ["mv",fileName.with_suffix(".temp"),fileName],check=True
    )
    return sparseMatrixDesc(fileName=fileName.__str__(),m=ms.m,n=ms.n,nnz=nnz)

def generateMat(m,n,nnz,metafileDir:Path,baseName:str) -> sparseMatrixDesc:
    if not randomGeneratorPath.exists():
        raise RuntimeError(f"Can't find random generator! {randomGeneratorPath}!")
    fileName = metafileDir.resolve()/f"{baseName}-{m}-{n}-{nnz}.bincsr"
    print(f"Generating {fileName.__str__()}")
    subprocess.run(
        args = [randomGeneratorPath.__str__(),str(m),str(n),str(nnz),"0",fileName.with_suffix(".temp").__str__()],check=True)
    subprocess.run(
        args = ["mv",fileName.with_suffix(".temp"),fileName],check=True
    )
    return sparseMatrixDesc(fileName=fileName.__str__(),m=m,n=n,nnz=nnz)


def generateGroupNew(mnnnzList:List[Tuple[MatrixSetting,MatrixSetting]],metafileDir:Path,baseName:str):
    metafileDir.mkdir(parents=True,exist_ok=True)
    metainfo = []
    for ma,mb in mnnnzList:
        metainfo.append({
                "matA":generateMatNew(ma,metafileDir,baseName+".matA").toDict(),
                "matB":generateMatNew(mb,metafileDir,baseName+".matB").toDict()
            }
        )
        # metainfo.append(generateMat(m,n,nnz,metafileDir,baseName).toDict())
        # metainfo.append(generateMatDesc(m,n,nnz,metafileDir,baseName).toDict())
    metaFilePath = metafileDir / matrixMetaFile(baseName)
    with metaFilePath.open("w") as f:
        json.dump(metainfo,f,indent=4)

def generateGroup(mnnnzList:List[Tuple[int,int,int]],metafileDir:Path,baseName:str):
    metafileDir.mkdir(parents=True,exist_ok=True)
    metainfo = []
    for m,n,nnz in mnnnzList:
        metainfo.append(generateMat(m,n,nnz,metafileDir,baseName).toDict())
        # metainfo.append(generateMatDesc(m,n,nnz,metafileDir,baseName).toDict())
    metaFilePath = metafileDir / matrixMetaFile(baseName)
    with metaFilePath.open("w") as f:
        json.dump(metainfo,f,indent=4)


################################################################
#               Generate workload
###############################################################


def loadMatrixDescListNew(metaFilePath: Path) -> List[Tuple[sparseMatrixDesc,sparseMatrixDesc]]:
    if not metaFilePath.exists():
        raise RuntimeError("Cannot open file" + str(Path))
    with metaFilePath.open("r") as f:
        jcontent = json.load(f)

    return [(sparseMatrixDesc.fromDict(d["matA"]),
            sparseMatrixDesc.fromDict(d["matB"]),
                ) for d in jcontent]

def loadMatrixDescList(metaFilePath: Path) -> List[sparseMatrixDesc]:
    if not metaFilePath.exists():
        raise RuntimeError("Cannot open file" + str(Path))
    with metaFilePath.open("r") as f:
        jcontent = json.load(f)

    return [sparseMatrixDesc.fromDict(d) for d in jcontent]


def generateSPGemmWorkload(abList: List[Tuple[sparseMatrixDesc, sparseMatrixDesc,str]],
                           metafileDir: Path, baseName: str):
    metafileDir = metafileDir.resolve()
    if not metafileDir.exists():
        metafileDir.mkdir(parents=True, exist_ok=True)
    
    metaInfoList = []
    for i,(a,b,kernelName) in enumerate(abList):
        workload = workloadDesc(
            kernelName = kernelName,
            inputs={
                "matrixA": a,
                "matrixB": b
            },
            outputs={
                "matrixC": sparseMatrixDesc(
                    fileName=(metafileDir/f"{baseName}-{i}-matrixC.bincsr").__str__()
                )
            }
        )
        workloadSpecPath = metafileDir / f"{baseName}-{i}-workload.json"
        print(f"Making workload {workloadSpecPath.__str__()}")
        with workloadSpecPath.open("w") as f:
            json.dump(workload.toDict(),f,indent=4)
        metaInfoList.append(workloadSpecPath.__str__())
    
    workeloadMetaInfoFilePath = metafileDir/ workloadMetaFile(baseName)
    with workeloadMetaInfoFilePath.open("w") as f:
        json.dump(metaInfoList,f,indent=4)
    


################################################################
#               Execute workloads
###############################################################



def runWorkloadGem5Arm(workloadDescFilePath:Path):
    import gem5ArmRunner
    import sys
    if not harnessPath.exists():
        raise RuntimeError("Can't find harness!")
    workloadFile = workloadDescFilePath.__str__()
    reportFile = getReportFilePath(workloadDescFilePath).__str__()
    scalar_repeat = 3
    simd_repeat = 3
    for e in sys.argv:
        if e.startswith("scalar="):
            scalar_repeat = int(e[len("scalar="):])
        if e.startswith("simd="):
            simd_repeat = int(e[len("simd="):])
    print(f"Executing harness: \nworkloadFile = {workloadFile}\nreportFile={reportFile} scalar_repeat={scalar_repeat} simd_repeat={simd_repeat}\n")
    outdir = workloadDescFilePath.with_suffix("")
    outdir.mkdir(parents=True,exist_ok=True)
    gem5ArmRunner.gem5Run(
        cwd=outdir,
        bin = harnessPath.__str__(),
        args=[workloadFile,reportFile,str(scalar_repeat),str(simd_repeat)],
        outdir=outdir
    )

def runWorkloadGem5ArmOld(workloadDescFilePath:Path):
    import gem5Exec
    if not harnessPath.exists():
        raise RuntimeError("Can't find harness!")
    workloadFile = workloadDescFilePath.__str__()
    reportFile = getReportFilePath(workloadDescFilePath).__str__()
    print(f"Executing harness: \nworkloadFile = {workloadFile}\nreportFile={reportFile}\n")
    outdir = workloadDescFilePath.with_suffix("")
    outdir.mkdir(parents=True,exist_ok=True)
    gem5Exec.gem5Run(
        cwd=outdir,
        bin = harnessPath.__str__(),
        args=[workloadFile,reportFile],
        outdir=outdir
    )
    # subprocess.run(
    #     args = [
    #         harnessPath.__str__(),
    #         workloadFile,
    #         reportFile
    #     ],check=True
    # )

def runWorkloadOnHost(workloadDescFilePath:Path):
    if not harnessPath.exists():
        raise RuntimeError("Can't find harness!")
    workloadFile = workloadDescFilePath.__str__()
    reportFile = getReportFilePath(workloadDescFilePath).__str__()
    print(f"Executing harness: \nworkloadFile = {workloadFile}\nreportFile={reportFile}\n")
    subprocess.run(
        args = [
            harnessPath.__str__(),
            workloadFile,
            reportFile
        ],check=True
    )

def runWorkload(workloadDescFilePath:Path,useGem5:bool = False):
    if useGem5:
        runWorkloadGem5Arm(workloadDescFilePath)
    else:
        runWorkloadOnHost(workloadDescFilePath)


def runAllWorkloadInDir(theDir:Path,useGem5:bool = False):
    allWorkload = theDir.glob("*-workload.json")
    allWorkload = [(theDir/wl).resolve() for wl in allWorkload]
    print(f"Will execute following workloads:")
    for wl in allWorkload:
        print(wl.__str__())
    
    for wl in allWorkload:
        runWorkload(wl,useGem5)

def runAllWorkloadInMeta(theMeta:Path,useGem5:bool = False,parallel=False) -> List[Thread]:
    with theMeta.open("r") as f:
        theList = json.load(f)
    allWorkload = [(theMeta.parent/st).resolve() for st in theList]
    print(f"Will execute following workloads:")
    for wl in allWorkload:
        print(wl.__str__())
    
    if not parallel:
        for wl in allWorkload:
            runWorkload(wl,useGem5)
        return []
    else:
        allThread = [] # type: List[Thread]
        for wl in allWorkload:
            t = Thread(target=runWorkload,args=(wl,useGem5))
            t.start()
            allThread.append(t)
        return allThread




################################################################
#               Gather report
###############################################################



def loadWorkLoadMetaList(metafilePath:Path):
    with metafilePath.open("r") as f:
        jcontent = json.load(f)
    return [metafilePath.parent/st for st in jcontent]

def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            for i,a in enumerate(x):
                flatten(a, name + str(i) + '_')
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def gatherReports(allReportPaths:List[Path],finalReportPath:Path,finalReportCSVPath:Path,filterName = lambda x: True) -> List[reportFile]:
    reportList = []
    for reportsPath in allReportPaths:
        if not reportsPath.exists():
            raise RuntimeError(f"File {reportsPath.__str__()} not exisit.")
        if reportsPath.is_dir():
            reportList += list(reportsPath.glob("*-report.json"))
        else:
            reportList += [getReportFilePath(p) for p in loadWorkLoadMetaList(reportsPath)]
        
    print(f"Gather information from following files:\n" +
        "\n".join([x.__str__() for x in reportList])
    )

    flatjsonList = []
    for r in reportList:
        with r.open("r") as f:
            j =  flatten_json(json.load(f))
            jf = {name:val for name,val in j.items() if filterName(name)}
            flatjsonList.append(jf)
    

    if len(reportList) == 0:
        return
    
    # csvPath.parent.mkdir(parents=True,exist_ok=True)
    # with csvPath.open("w") as f:
    #     writer = csv.writer(f)
        
    #     keys0 = flatjsonList[0].keys()
    #     writer.writerow(["fileName"] + list(keys0))

    #     for fn,jf in zip(reportList,flatjsonList):
    #         writer.writerow([fn.name] + list(jf.values()))
    finalReportPath.parent.mkdir(parents=True,exist_ok=True)
    with finalReportPath.open("w") as f:
        json.dump(flatjsonList,f,indent=4)
    
    df = pd.DataFrame(flatjsonList)
    dfSelect = df[["kernelName","inputs_matrixA_edgeFactor","timmer_SpGemm_mean"]]
    tb = dfSelect.groupby(["kernelName","inputs_matrixA_edgeFactor"]).sum().unstack("inputs_matrixA_edgeFactor")
    tb.columns = tb.columns.get_level_values(1)
    tb.reset_index()
    # tb.to_clipboard()
    finalReportCSVPath.parent.mkdir(parents=True,exist_ok=True)
    with finalReportCSVPath.open("w") as f:
        tb.to_csv(f)
