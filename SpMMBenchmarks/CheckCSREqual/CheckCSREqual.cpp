#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "SparseMatTool/format.hpp"
#include "SparseMatTool/serialization.hpp"



int main(int argv, char **argc)
{
    if (argv != 3)
    {
        printf("Usage: %s fileName1 fileName2\n", argc[0]);
        return -1;
    }

    CSR a, b;
    a = loadCSR(argc[1]);
    b = loadCSR(argc[2]);

    if (a.m != b.m || a.n != b.n || a.nnz != b.nnz)
    {
        printf("Shape is not equal:\n a: m=%lu, n=%lu, nnz=%lu\n b: m=%lu, n=%lu, nnz=%lu\n", a.m, a.n, a.nnz, b.m, b.n, b.nnz);
        return 1;
    }

    if(a.rowBeginOffset != b.rowBeginOffset){
        printf("A's row begin offset is not the same as B's\n");
        return -1;
    }

    if(a.colIdx != b.colIdx){
        printf("A's colIdx is not the same as B's\n");
        return -1;
    }

    if(a.values.size() != b.values.size()){
        printf("A's value's size is not the same as B's\n");
        return -1;
    }

    for(Size_t i = 0; i < a.values.size(); ++i){
        if(std::abs(a.values[i] - b.values[i]) > 1e-2){
            printf("A's value is not the same as B's\n");
        }
    }
    
    printf("Pass: Results are equal\n");
    return 0;

}