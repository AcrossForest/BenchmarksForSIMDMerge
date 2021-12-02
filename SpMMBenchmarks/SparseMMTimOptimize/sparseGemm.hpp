#pragma once
#include <algorithm>
#include "SparseMatTool/format.hpp"
#include "rowStack.hpp"
#include "Benchmarking/benchmarking.hpp"
#include "scalarSparseAdd.hpp"
#include "simdSparseAdd.hpp"
#include "fixGem5Bug.hpp"
#include "measureMerge.hpp"

namespace OptimizedTim{
template<class Toolkit>
CSR SpMM_TimSortOptimized(TimmerHelper& timmerHelper,Toolkit toolkit,const CSR &a, const CSR &b)
{
    // c_{m,n} = a_{m,k} * b_{k,n}
    Idx m, k, n;
    m = a.m;
    k = a.n;
    n = b.n;

    // Estimate the size of output matrix
    Size_t nnzCMax = 0;
    Size_t nnzCRowMax = 0;
    NNZIdx nnzARowMax = 0;
    for (Idx r = 0; r < m; ++r)
    {
        // The nnz of C[r,:]
        Size_t nnzCr = 0;
        for (Size_t off = a.rowBeginOffset[r]; off < a.rowBeginOffset[r + 1]; ++off)
        {
            Idx a_col = a.colIdx[off];
            nnzCr += b.rowBeginOffset[a_col + 1] - b.rowBeginOffset[a_col];
        }
        // truncate by n
        nnzCr = std::min<Size_t>(nnzCr, n);

        nnzCMax += nnzCr;
        nnzCRowMax = std::max<Size_t>(nnzCRowMax, nnzCr);
        nnzARowMax = std::max<Idx>(nnzARowMax, a.rowBeginOffset[r + 1] - a.rowBeginOffset[r]);
    }

    CSR c;
    c.m = m;
    c.n = n;
    c.rowBeginOffset.resize(by64(m + 1));
    c.colIdx.resize(by64(nnzCMax));
    c.values.resize(by64(nnzCMax));
    Size_t nnzCReal = 0;

    auto stack = SparseRowStack(Toolkit{});

    stack.reserve(nnzCRowMax,b.colIdx.data(), b.values.data());
    timmerHelper.start();
    MergeClearTimmer();
    MergeStartTimmer(total);

    for (Idx r = 0; r < m; ++r)
    {

        c.rowBeginOffset[r] = nnzCReal;
        Size_t a_rowBegin = a.rowBeginOffset[r];
        NNZIdx a_rowNNZ = a.rowBeginOffset[r + 1] - a.rowBeginOffset[r];

        for (NNZIdx a_col_c_0 = 0; a_col_c_0 < a_rowNNZ; ++a_col_c_0)
        {
            NNZIdx a_col_c_off = a_col_c_0 + a_rowBegin;
            Idx a_col = a.colIdx[a_col_c_off];
            Val a_val = a.values[a_col_c_off];
            Idx b_row = a_col;
            Size_t b_rowBegin = b.rowBeginOffset[b_row];
            Size_t b_rowEnd = b.rowBeginOffset[b_row + 1];

            stack.putOnTop(a_val, b_rowBegin, b_rowEnd - b_rowBegin);
            stack.reduce();

        }

        stack.reduce(true,2);
        auto c_len = stack.exportResultTo(c.colIdx.data() + nnzCReal, c.values.data() + nnzCReal);
        nnzCReal += c_len;
        stack.clear();
    }
    MergeRecordTimmer(total);
    timmerHelper.end();

#if (defined __Measure_Merge__)
        std::cout
#define D(x) MergeDisplayTimmer(x)
MergeTimmers
#undef D
            << std::endl;
        

#endif//__Measure_Merge__

    c.nnz = nnzCReal;
    c.rowBeginOffset.resize(c.m+1);
    c.rowBeginOffset.back() = nnzCReal;
    c.colIdx.resize(nnzCReal);
    // c.colIdx.shrink_to_fit();
    c.values.resize(nnzCReal);
    // c.values.shrink_to_fit();

    return c;
}
}