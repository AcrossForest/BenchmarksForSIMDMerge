#!/usr/bin/env python3
import subprocess
from pathlib import Path
import sys
import os

try:
  GEM5_BUILD_PATH = os.environ['GEM5_BUILD_PATH']
  GEM5_SRC_PATH = os.environ['GEM5_SRC_PATH']
except KeyError:
  print("Please provide environment variable GEM5PATH (the path to gem5.opt) and GEM5SEPATH (the path to gem5's script se.py)")
  exit(-1)

vecLen = int(sys.argv[1])
KeyCombineLat = 3
MatchLat = 1
SEPermuteLat = 1

# Optimism latency
# KeyCombineLat = 1
# MatchLat = 0.5
# SEPermuteLat = 0.5

def run(name:str,*options:str):
  binPath = Path(__file__).parent / "build" / name

  subprocess.run(
    args=[
      Path(GEM5_BUILD_PATH) / "ARM/gem5.opt",
      Path(GEM5_SRC_PATH) / "configs/example/se.py",
      # f"${GEM5_BUILD_PATH}/ARM/gem5.opt",
      # f"${GEM5_SRC_PATH}/configs/example/se.py",
      "--caches", "--l2cache", 
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
binPath.absolute(),
"-o",
" ".join(options),
    ],
  )

if __name__ == "__main__":
  import sys
  run(sys.argv[2],*sys.argv[3:])