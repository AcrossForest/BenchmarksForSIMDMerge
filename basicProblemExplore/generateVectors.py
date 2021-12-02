#!/usr/bin/env python3
import numpy as np

def genRandVector(length:int,fileName:str):
  vec = np.random.randint(0,length*4,length,dtype=np.uint32)
  sortedVec = np.sort(vec)
  with open(fileName,"wb") as f:
    f.write(length.to_bytes(4,byteorder="little",signed=False))
    sortedVec.tofile(f,format="%d")
    



if __name__ == "__main__":
    import sys
    genRandVector(int(sys.argv[1]),sys.argv[2])