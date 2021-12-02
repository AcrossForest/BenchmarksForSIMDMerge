// Only intended for error checking and early stage debug
#pragma once
#include "SparseMatTool/format.hpp"
#include <unordered_set>

inline bool checkIfSparsePatternCorrect(
    const CSR &a, const CSR &b, Idx r,
    Idx *result_start, Idx *result_end)
{
    Size_t finalNNZ = result_end - result_start;

    std::unordered_set<Idx> expected, result;

    expected.reserve(finalNNZ);
    Size_t a_rowBegin = a.rowBeginOffset[r];
    Size_t a_rowEnd = a.rowBeginOffset[r + 1];
    NNZIdx a_rowNNZ = a_rowEnd - a_rowBegin;
    for (NNZIdx a_col_c_0 = 0; a_col_c_0 < a_rowNNZ; ++a_col_c_0)
    {
        Size_t a_col_c_off = a_col_c_0 + a_rowBegin;
        Idx a_col = a.colIdx[a_col_c_off];
        Idx b_row = a_col;

        Size_t b_rowBegin = b.rowBeginOffset[b_row];
        Size_t b_rowEnd = b.rowBeginOffset[b_row + 1];
        for (Size_t b_col_c_off = b_rowBegin; b_col_c_off < b_rowEnd; ++b_col_c_off)
        {
            Idx b_col = b.colIdx[b_col_c_off];
            expected.insert(b_col);
        }
    }

    result.insert(result_start,result_end);

    return expected == result;
}