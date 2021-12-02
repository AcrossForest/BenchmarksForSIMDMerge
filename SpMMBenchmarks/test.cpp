#include <string>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <random>
#include "SparseMatTool/format.hpp"
#include "SparseMatTool/serialization.hpp"
#include "SparseOldTranspose/SparseOldTranspose.hpp"
#include "Benchmarking/benchmarking.hpp"

int main()
{
    std::string file = "/root/source/performance/SparseMat/workload/big/tim/big-3-matrixC.bincsr";
    CSR a;

    Timmer timmer;
    timmer.measure("loadCSR", 0, 1, [&]() {
        a = loadCSR(file);
    });

    std::mt19937_64 gen(0);

    timmer.measure("shuffle", 0, 1, [&]() {
        for (Idx row = 0; row < a.m; ++row)
        {
            Size_t rowBegin = a.rowBeginOffset[row];
            Size_t rowEnd = a.rowBeginOffset[row + 1];
            std::shuffle(a.colIdx.begin() + rowBegin,
                                a.colIdx.begin() + rowEnd,gen);
        }
    });
    CSR doubleTranspose;
    CSR xsort;
    CSR trueSort;

    timmer.measure("doubleTrans", 1, 5, [&]() {
        doubleTranspose = oldTranspose(oldTranspose(a));
    });
    timmer.measure("xSortAllRow", 1, 5, [&]() {
        xsort = sortAllRow(a);
    });
    timmer.measure("perRowSort", 1, 5, [&]() {
        trueSort = sortAllRowDebug(a);
    });
    if(doubleTranspose != xsort){
        printf("Double transpose do not equal xsort\n");
    }
    if(trueSort != xsort){
        printf("True sort not equals xsort");
    }
    timmer.dump(true);
}