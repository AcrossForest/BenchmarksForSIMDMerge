#pragma once
#include <algorithm>
#include "SparseMatTool/format.hpp"
#include "SparseMMTimSortAlike/SparseRowStack.hpp"
#include "VectorUtils.hpp"
#include <iostream>
#include "M5MagicInst/m5ops.h"
#include "timmingDebug.hpp"
#include "Benchmarking/benchmarking.hpp"
Size_t instBasedVectorAdd(
    const Idx *__restrict a_pidx, const Val *__restrict a_pval, Size_t a_len,
    const Idx *__restrict b_pidx, const Val *__restrict b_pval, Size_t b_len,
    Idx *__restrict c_pidx, Val *__restrict c_pval);

Size_t directVectorAdd(Idx *__restrict a_pidx, Val *__restrict a_pval, Size_t a_len,
              Idx *__restrict b_pidx, Val *__restrict b_pval, Size_t b_len,
              Idx *__restrict c_pidx, Val *__restrict c_pval);
// #include "SparseMMTimSortAlike/CheckIfSparsePatternCorrect.hpp"

template<class Callable>
CSR SpMM_TimSortAlike(TimmerHelper& timmerHelper,Callable merge,const CSR &a, const CSR &b)
{

#if (defined __UseTimmingDebug__)
#define D(x) ClearTimmer(x)
MoreTimmers
#undef D
#endif

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

    SparseRowStack stack(merge);
    stack.reserve(nnzARowMax, nnzCRowMax);
    timmerHelper.start();

    m5_dump_reset_stats(0,0);

    

    for (Idx r = 0; r < m; ++r)
    {
        StartTimmer(total);

        StartTimmer(outLoop1);
        c.rowBeginOffset[r] = nnzCReal;
        Size_t a_rowBegin = a.rowBeginOffset[r];
        NNZIdx a_rowNNZ = a.rowBeginOffset[r + 1] - a.rowBeginOffset[r];
        RecordTimmer(outLoop1);
        if (a_rowNNZ == 0){
            RecordTimmer(total);
            continue;
        }
        
        StartTimmer(mainLoop);
        for (NNZIdx a_col_c_0 = 0; a_col_c_0 < a_rowNNZ; ++a_col_c_0)
        {
            StartTimmer(loopPart1);
            NNZIdx a_col_c_off = a_col_c_0 + a_rowBegin;
            Idx a_col = a.colIdx[a_col_c_off];
            Val a_val = a.values[a_col_c_off];
            Idx b_row = a_col;
            Size_t b_rowBegin = b.rowBeginOffset[b_row];
            Size_t b_rowEnd = b.rowBeginOffset[b_row + 1];

            stack.allocateOnTop(b_rowEnd - b_rowBegin);
            auto info = stack.getTop(1);

            RecordTimmer(loopPart1);

            StartTimmer(loopPartMovMul);
            vectorMoveMul(
                b.colIdx.data() + b_rowBegin, b.values.data() + b_rowBegin, a_val,
                b_rowEnd - b_rowBegin,
                stack.idxStack[info.pos].data() + info.start,
                stack.valueStack[info.pos].data() + info.start
            );
            RecordTimmer(loopPartMovMul);

            // std::copy(b.colIdx.data() + b_rowBegin, b.colIdx.data() + b_rowEnd,
            //           stack.idxStack.data() + info.start);
            // std::transform(
            //     b.values.data() + b_rowBegin,
            //     b.values.data() + b_rowEnd,
            //     stack.valueStack.data() + info.start,
            //     [a_val](Val v) {
            //         return v * a_val;
            //     });

            StartTimmer(loopPartReduce);
            stack.reduce();
            RecordTimmer(loopPartReduce);
        }
        RecordTimmer(mainLoop);

#ifdef __OptimizeFinalMove__
        StartTimmer(finalReduce);
        stack.reduce(true,2);
        RecordTimmer(finalReduce);

        StartTimmer(copyBack);
        auto c_len = stack.exportResultTo(c.colIdx.data() + nnzCReal, c.values.data() + nnzCReal);
        nnzCReal += c_len;
        RecordTimmer(copyBack);
#else
        // Reduce to only one vector
        StartTimmer(finalReduce);
        stack.reduce(true);
        RecordTimmer(finalReduce);

        // Now, copy the result out
        StartTimmer(copyBack);
        auto info = stack.getTop(1);
        vectorCopy(
            stack.idxStack[info.pos].data() + info.start,
            stack.valueStack[info.pos].data() + info.start,
            info.length,
            c.colIdx.data() + nnzCReal,c.values.data() + nnzCReal
        );

        // std::copy(stack.idxStack.data() + info.start,
        //           stack.idxStack.data() + info.start + info.length,
        //           c.colIdx.data() + nnzCReal);
        // std::copy(stack.valueStack.data() + info.start,
        //           stack.valueStack.data() + info.start + info.length,
        //           c.values.data() + nnzCReal);
        nnzCReal += info.length;
        RecordTimmer(copyBack);
#endif

        ////////////////////////////////////////////
        // Only for debug
        // bool check = checkIfSparsePatternCorrect(
        //     a, b, r,
        //     stack.idxStack.data() + info.start,
        //     stack.idxStack.data() + info.start + info.length);
        // if(!check) {
        //     throw std::logic_error("The result sparsity pattern is in correct.");
        // }
        ///////////////////////////////////////

        StartTimmer(cleanStack);
        stack.clear();
        RecordTimmer(cleanStack);

#if (defined __UseTimmingDebug__)
    StartTimmer(print);
    if (r % 2 == 0){
        // std::cout << "At row: " << r << "\t loopCount: " << totalLoopcount
        //     << "\t processed: " << totalProccessed
        //     << "\t Efficiency: " << double(totalProccessed)/double(totalLoopcount)
        //     << std::endl;
        std::cout
#define D(x) DisplayTimmer(x)
MoreTimmers
#undef D
            << std::endl;
        }
    RecordTimmer(print);
#endif

    RecordTimmer(total);

    }





    m5_dump_reset_stats(0,0);
    timmerHelper.end();

    c.nnz = nnzCReal;
    c.rowBeginOffset.resize(c.m+1);
    c.rowBeginOffset.back() = nnzCReal;
    c.colIdx.resize(nnzCReal);
    // c.colIdx.shrink_to_fit();
    c.values.resize(nnzCReal);
    // c.values.shrink_to_fit();

    return c;
}
