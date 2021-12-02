#!/bin/bash


## Bérenger Bramas (berenger.bramas@inria.fr)
## ARM SVE SORT
## This file build the different binaries to test
## the ARM SVE SORT.
## Using the default one (without passing any definition
## should provide the best binary)

if [[ "$CXX" == "" ]] ; then
    CXX=armclang++
fi
echo "[CONF] Compiler is: $CXX (set CXX variable to change that)"

if [[ "$MARCH" == "" ]] ; then
    MARCH="armv8.2-a+sve"
fi
echo "[CONF] arch is: $MARCH (set MARCH variable to change that)"

if [[ "$CXXFLAGS" == "" ]] ; then
    CXXFLAGS="-O3 -DNDEBUG"
fi
echo "[CONF] extra flags: $CXXFLAGS (set CXXFLAGS variable to change that)"


FULLCXXFLAGS="$CXXFLAGS"


echo "FULLCXXFLAGS = $FULLCXXFLAGS"

echo "$CXX -DNDEBUG $FULLCXXFLAGS -march=$MARCH -fopenmp sortSVEperf.cpp -o sortSVEperf.$CXX.exe"
$CXX $FULLCXXFLAGS "-march=$MARCH" -fopenmp sortSVEperf.cpp -o "sortSVEperf.$CXX.exe"

echo "$CXX -DNDEBUG $FULLCXXFLAGS -march=$MARCH -fopenmp sortSVEtest.cpp -o sortSVEtest.$CXX.exe"
$CXX $FULLCXXFLAGS "-march=$MARCH" -fopenmp sortSVEtest.cpp -o "sortSVEtest.$CXX.exe"

