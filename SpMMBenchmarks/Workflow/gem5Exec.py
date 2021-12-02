import subprocess
from pathlib import Path
from typing import List
gem5Obj = "/home/stoic_ren/source/SpSpProject/gem5/build/ARM/gem5.opt"
script = "/home/stoic_ren/source/SpSpProject/gem5/configs/example/se.py"
# bin = "/root/source/testSpSpInst/build/SparseVectorAdd/SparseVectorAdd"

param = []
def addParams(oneParam:str):
  param.append("--param")
  param.append(oneParam)

addParams('system.cpu[:].isa[:].sve_vl_se = 16')
# addParams('system.cpu[:].dcache.prefetcher.degree = 4')
# addParams 'system.mem_ctrls[:].bandwidth=\'32.8GB/s\''
# addParams 'system.mem_ctrls[:].latency=\'3ns\''
addParams('system.cpu[:].dcache.size=\'16MB\'')
# addParams 'system.tol2bus.width=128'
# addParams 'system.l2.response_latency=1'
# addParams 'system.l2.data_latency=1'
# addParams 'system.l2.tag_latency=1'

addParams('system.cpu[:].dcache.prefetch_on_access=True')


gem5Config = []

hwConfigs = [
  "--caches", "--l2cache", "--mem-type","SimpleMemory",
  "--cacheline_size","64",
  *param,
  "--cpu-type","O3_ARM_v7a_3",
  "--l1d-hwp-type","TaggedPrefetcher"
]

# set hwConfigs \
#     --caches --l2cache --mem-type SimpleMemory \
#     --cacheline_size 64 \
#     $param \
#     --cpu-type O3_ARM_v7a_3 \
#     --l1d-hwp-type TaggedPrefetcher
#     # --l1d-hwp-type IndirectMemoryPrefetcher

simConfigs = []


def gem5Run(cwd:Path,bin:str,args:List[str],outdir:Path):
  print( " ".join([
          gem5Obj,
          *gem5Config,
          script,
          *hwConfigs,
          *simConfigs,
          "--cmd",bin,
          "-o", " ".join(args)
        ]))
  subprocess.run(
        args = [
          gem5Obj,
          *gem5Config,
          "-d",outdir.__str__(),
          script,
          *hwConfigs,
          *simConfigs,
          "--cmd",bin,
          "-o", " ".join(args)
        ],check=True,cwd = cwd
    )


# # set simConfigs \

# function run
#     /root/source/gem5/build/ARM/gem5.opt \
#         $gem5Config \
#         $script \
#         $hwConfigs \
#         $simConfigs \
#         --cmd $bin
# end




