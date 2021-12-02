#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import sys
from typing import List

try:
  GEM5_BUILD_PATH = os.environ['GEM5_BUILD_PATH']
  GEM5_SRC_PATH = os.environ['GEM5_SRC_PATH']
except KeyError:
  print("Please provide environment variable GEM5PATH (the path to gem5.opt) and GEM5SEPATH (the path to gem5's script se.py)")
  exit(-1)

# vecLen = int(sys.argv[1])
KeyCombineLat = 3
MatchLat = 1
SEPermuteLat = 1

# # Optimism latency
# KeyCombineLat = 0.5
# MatchLat = 0.5
# SEPermuteLat = 0.5

# def gem5Run(cwd:Path,bin:str,args:List[str],outdir:Path):
#   print( " ".join([
#           "/root/source/gem5/build/ARM/gem5.opt",
#           *gem5Config,
#           script,
#           *hwConfigs,
#           *simConfigs,
#           "--cmd",bin,
#           "-o", " ".join(args)
#         ]))
#   subprocess.run(
#         args = [
#           "/root/source/gem5/build/ARM/gem5.opt",
#           *gem5Config,
#           "-d",outdir.__str__(),
#           script,
#           *hwConfigs,
#           *simConfigs,
#           "--cmd",bin,
#           "-o", " ".join(args)
#         ],check=True,cwd = cwd
#     )

def gem5Run(cwd:Path ,bin:str,args:List[str],outdir:Path):
  # binPath = Path(__file__).parent / "build" / name
  vecLen = 4

  for e in sys.argv:
    if e.startswith("vecLen="):
      vecLen = int(e[len("vecLen="):])
  

  toRun = [
      Path(GEM5_BUILD_PATH) / "ARM/gem5.opt",
      "-d",outdir.__str__(),
      Path(GEM5_SRC_PATH) / "configs/example/se.py",
      # "/root/source/gem5GCC10/build/ARM/gem5.opt",
      # "/root/source/gem5GCC10/configs/example/se.py",
      "--caches", "--l2cache", 
      # "--l1d_size","16MB", # test if it is bounded by L1 <-> L2 bandwidth. Don't set it in normal expriment.
      "--mem-type", "SimpleMemory",
      "--cacheline_size", "64",
      "--cpu-type", "O3_ARM_v7a_3",
      "--l1d-hwp-type", "TaggedPrefetcher",
      "--mem-size","2GB",

"--param",
f'''
system.cpu[:].dcache.prefetcher.degree = 4
system.cpu[:].dcache.prefetch_on_access=True
system.cpu[:].isa[:].sve_vl_se = {vecLen} / 4
from common.SpSpLatency import *
resetSpSpLatency(system.cpu[0].fuPool.FUList[4],SpSpLatencySetting(vecLen={vecLen},KeyCombineLat={KeyCombineLat},MatchLat={MatchLat},SEPermuteLat={SEPermuteLat}))
''',

"--cmd",
bin,
"-o", " ".join(args)
    ]
  print(toRun)

  subprocess.run(
    args=toRun,check=True,cwd = cwd
  )
