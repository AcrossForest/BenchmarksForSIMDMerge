[![pipeline status](https://gitlab.inria.fr/bramas/arm-sve-sort/badges/master/pipeline.svg)](https://gitlab.inria.fr/bramas/arm-sve-sort/commits/master)

[![coverage report](https://gitlab.inria.fr/bramas/arm-sve-sort/badges/master/coverage.svg)](https://bramas.gitlabpages.inria.fr/arm-sve-sort/)

Berenger Bramas - Inria - berenger.bramas@inria.fr

Functions to sorts arrays of integers or doubles, or arrays of keys/values of integers (in the format of two arrays of integers or one array of pairs of integers) using the ARM SVE intrinsics.
The code provides sort functions to small arrays (<= 16 SVE vectors), for larger arrays, for partitioning arrays, or to sort in parallel.
The code also have two versions, one is following the SVE phylosophy by working for any vector size (known at runtime), whereas the second version (postfix "512") works only if the vectors are of size 512bits.
This 512bits version is a pure copy of the AVX512 sort (code: https://gitlab.inria.fr/bramas/avx-512-sort paper: http://thesai.org/Publications/ViewPaper?Volume=8&Issue=10&Code=IJACSA&SerialNo=44).

The paper that describes the current ARM SVE sort is available as a preprint at: https://hal.inria.fr/hal-03227631

# What if you do not have SVE but would like to test the code

CPUs with ARM SVE feature are rare. Therefore, if you would like to compile a code with SVE intrinsics you can use the scalar and portable implementation of SVE https://gitlab.inria.fr/bramas/farm-sve .
In the current code, you will have to rename `farm_sve.h` into `arm_sve.h` (or create a link) and include the directory (usually `-I.` if the file is located in the current directory).
Of course, the performance will mean nothing, but that will allow to test the code or update it without having the right CPU.
The CI file `.gitlab-ci.yml` uses this mechanism.

# Compilation

The headers have no dependencies but needs OpenMP to provide the parallel sorting functions.
A script named `build-script.sh` can be used to generate the binaries.

Building by hand can be done with:
```bash
# Set correct arch and compiler
MARCH="armv8.2-a"
CXX=armclang++
# Build
$CXX -DNDEBUG -O3 -march=$MARCH+sve -fopenmp sortSVEperf.cpp -o sortSVEperf.exe
$CXX -DNDEBUG -O3 -march=$MARCH+sve -fopenmp sortSVEtest.cpp -o sortSVEtest.exe
```

## Supported compilers

- arm/20.3 (armclang++): works
- gcc/11.1.0 (g++) : works (with version 11-20210321 I cannot compile the backend complains about unknown instructions and crashes)
- fujitsu-compiler/4.3.1 (FCC): cannot compile (internal compiler error with old code version, and unsupported tuple management with new code version)
- cce-sve/10.0.1 (CC): cannot compile "Unable to find cray-libsci/20.09.1.1 libraries compatible with cce/10.0.1." but cce is there! Note that the arch selection is different (-hcpu=native)

# List of functions

For namespace `SortSVE` (sortSVE.hpp) and `SortSVE512` (sortSVE512.hpp):
- SortSVE::Sort(); to sort an array
- SortSVE::SortOmpPartition(); to sort in parallel
- SortSVE::SortOmpMerge(); to sort in parallel
- SortSVE::SortOmpMergeDeps(); to sort in parallel (advised version)
- SortSVE::SortOmpParMerge(); to sort in parallel
- SortSVE::PartitionSVE(); to partition
- SortSVE::SmallSort16V(); to sort a small array (should be less than 16 AVX512 vectors)
Each function supports `int` and `double`.

For namespace `SortSVEkv` (sortSVEkv.hpp) and `SortSVEkv512` (sortSVEkv512.hpp):
- SortSVE::Sort(); to sort an array
- SortSVE::PartitionSVE(); to partition
- SortSVE::SmallSort16V(); to sort a small array (should be less than 16 AVX512 vectors)
Each function supports `int` for both the keys and the values.
Note that `SortSVEkv` supports both two arrays for keys/values or one array of pairs.

# Citing

Refer to the preprint https://hal.inria.fr/hal-03227631:
```
@unpublished{bramas:hal-03227631,
  TITLE = {{A fast vectorized sorting implementation based on the ARM scalable vector extension (SVE)}},
  AUTHOR = {Bramas, B{\'e}renger},
  URL = {https://hal.inria.fr/hal-03227631},
  NOTE = {working paper or preprint},
  YEAR = {2021},
  MONTH = May,
  PDF = {https://hal.inria.fr/hal-03227631/file/svesort.pdf},
  HAL_ID = {hal-03227631},
  HAL_VERSION = {v1},
}
```

