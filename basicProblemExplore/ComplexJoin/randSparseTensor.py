#! /usr/bin/env python3
from os import write
from typing import IO
import sparse as sp
import numpy as np
import sys
def dumpRandomSparseTensor(f:IO,mode,valNum,nnz,dims,seed):

    spTensor = sp.asCOO(sp.random(dims,nnz=nnz,random_state=seed))
    vals = np.random.rand(valNum,nnz).astype(np.float32)

    print(f"real shape is : {spTensor.coords.shape}\n")

    def writeInt(a:int):
        f.write(a.to_bytes(4,"little"))
    
    writeInt(741) # magic number
    writeInt(mode)
    writeInt(valNum)
    writeInt(nnz)

    for m in range(mode):
        f.write(
            np.array(spTensor.coords[m,:],dtype=np.int32).tobytes()
        )
    for v in range(valNum):
        f.write(vals[v,:].tobytes())

if __name__=="__main__":
    [_,mode,valNum,nnz,density,fileName] = sys.argv
    mode = int(mode)
    valNum = int(valNum)
    nnz = int(nnz)
    density = float(density)

    dims = [int((nnz/density)**(1/mode))] * mode

    print(f"mode = {mode}, nnz = {nnz}, density = {density}, --> dims = {dims}")



    with open(fileName,"wb") as f:
        dumpRandomSparseTensor(f,mode=mode,valNum=valNum,nnz=nnz,dims=dims,seed=0)
        # print(f.tell())
        dumpRandomSparseTensor(f,mode=mode,valNum=valNum,nnz=nnz,dims=dims,seed=1)
        # print(f.tell())
        


