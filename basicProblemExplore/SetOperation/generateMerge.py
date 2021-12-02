#!/usr/bin/env python3
from sys import argv
import numpy as np
from typing import IO
from sys import argv

def randVector(f:IO,l:int):
  vec = np.random.randint(0,10**8,l,dtype=np.uint32)
  vec:np.ndarray = np.sort(vec)
  f.write(l.to_bytes(4,"little"))
  f.write(vec.tobytes())

def dumpVector(f:IO, arr:np.ndarray):
  length:int = int(arr.size)
  f.write(length.to_bytes(4,"little"))
  f.write(arr.tobytes())

if __name__ == "__main__":
  (cmd,fileName,len1,len2,overlapNum) = argv
  len1,len2,overlapNum = int(len1), int(len2), int(overlapNum)
  a_and_b = np.random.randint(0,10**8,overlapNum,dtype=np.uint32)
  a_only = np.random.randint(0,10**8,len1-overlapNum,dtype=np.uint32)
  b_only = np.random.randint(0,10**8,len2-overlapNum,dtype=np.uint32)

  vec_a = np.sort(np.unique(np.concatenate([a_and_b,a_only])))
  vec_b = np.sort(np.unique(np.concatenate([a_and_b,b_only])))
  with open(fileName,"wb") as f:
    dumpVector(f,vec_a)
    dumpVector(f,vec_b)
  
