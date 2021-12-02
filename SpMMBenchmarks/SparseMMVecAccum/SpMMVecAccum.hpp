#pragma once
#include <algorithm>
#include <memory>
#include "SparseMatTool/format.hpp"
#include "SparseOldTranspose/SparseOldTranspose.hpp"
#include "Benchmarking/benchmarking.hpp"
#include <unordered_map>

enum class SpMMDenseSort{
    row_quicksort,
    no_sort,
    mat_linklist
};

template <SpMMDenseSort>
struct SpMMSortTag{};

template <SpMMDenseSort method>
CSR SpMM_DenseVec(TimmerHelper& timmerHelper,SpMMSortTag<method>,const CSR &a, const CSR &b)
{
    // c_{m,n} = a_{m,k} * b_{k,n}
    Idx m, k, n;
    m = a.m;
    k = a.n;
    n = b.n;



    // Estimate the size of output matrix
    Size_t nnzCMax = 0;
    for (Idx r = 0; r < m; ++r)
    {
        // The nnz of C[r,:]
        Size_t nnzCr = 0;
        for (Size_t off = a.rowBeginOffset[r]; off < a.rowBeginOffset[r + 1]; ++off)
        {
            Idx a_col = a.colIdx[off];
            nnzCr += b.rowBeginOffset[a_col + 1] - b.rowBeginOffset[a_col];
        }
        nnzCMax += std::min<Size_t>(nnzCr, n);
    }

    CSR c;
    c.m = m;
    c.n = n;
    c.rowBeginOffset.resize(m + 1);
    c.colIdx.resize(nnzCMax);
    c.values.resize(nnzCMax);

    Size_t matrixNnz = 0;
    std::unique_ptr<Idx[]> nonZeroCol = std::make_unique<Idx[]>(n);
    std::unique_ptr<bool[]> bitmat = std::make_unique<bool[]>(n);
    std::unique_ptr<Val[]> valArray = std::make_unique<Val[]>(n);
    // Buffer<Idx> nonZeroCol(n);
    // Buffer<bool> bitmat(n, false);
    // Buffer<Val> valArray(n, 0);

    timmerHelper.start();
    for(Idx i=0; i<n; ++i){
        bitmat[i] = false;
        valArray[i] = 0;
    }

    for (Idx r = 0; r < m; ++r)
    {
        NNZIdx rowNnz = 0;
        for (Size_t off = a.rowBeginOffset[r]; off < a.rowBeginOffset[r + 1]; ++off)
        {
            Idx a_col = a.colIdx[off];
            Val a_val = a.values[off];
            for (Size_t offB = b.rowBeginOffset[a_col]; offB < b.rowBeginOffset[a_col + 1]; ++offB)
            {
                Idx b_col = b.colIdx[offB];
                Val b_val = b.values[offB];

                if (!bitmat[b_col])
                    nonZeroCol[rowNnz++] = b_col;
                bitmat[b_col] = true;
                valArray[b_col] += a_val * b_val;
            }
        }
        if constexpr (method == SpMMDenseSort::row_quicksort){
            std::sort(nonZeroCol.get(), nonZeroCol.get() + rowNnz);
        }

        c.rowBeginOffset[r] = matrixNnz;
        for (NNZIdx i = 0; i < rowNnz; ++i)
        {
            Size_t pos = matrixNnz + i;
            c.colIdx[pos] = nonZeroCol[i];
            c.values[pos] = valArray[nonZeroCol[i]];

            bitmat[nonZeroCol[i]] = false;
            valArray[nonZeroCol[i]] = 0;
        }
        matrixNnz += rowNnz;
        rowNnz = 0;
    }

    c.nnz = matrixNnz;
    c.rowBeginOffset.back() = matrixNnz;

    if constexpr (method == SpMMDenseSort::row_quicksort 
            || method == SpMMDenseSort::no_sort){
        timmerHelper.end();
        c.colIdx.resize(matrixNnz);
        c.colIdx.shrink_to_fit();
        c.values.resize(matrixNnz);
        c.values.shrink_to_fit();
        return c;
    } else {
        // method == SpMMDenseSort::mat_linklist
        auto temp =  sortAllRow(c);
        timmerHelper.end();
        return temp;
    }
}


template <SpMMDenseSort method>
CSR SpMM_Hash(TimmerHelper& timmerHelper,SpMMSortTag<method>,const CSR &a, const CSR &b)
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
    c.rowBeginOffset.resize(m + 1);
    c.colIdx.resize(nnzCMax);
    c.values.resize(nnzCMax);

    Size_t matrixNnz = 0;

    std::unordered_map<Idx,Val> themap;
    themap.reserve(2*nnzCRowMax);
    std::unique_ptr<std::pair<Idx,Val>[]> sortBuffer= std::make_unique<std::pair<Idx,Val>[]>(nnzCRowMax);
    // std::unique_ptr<Idx[]> nonZeroCol = std::make_unique<Idx[]>(n);
    // std::unique_ptr<bool[]> bitmat = std::make_unique<bool[]>(n);
    // std::unique_ptr<Val> valArray = std::make_unique<Val[]>(n);
    // Buffer<Idx> nonZeroCol(n);
    // Buffer<bool> bitmat(n, false);
    // Buffer<Val> valArray(n, 0);

    timmerHelper.start();
    // for(Idx i=0; i<n; ++i){
    //     bitmat[i] = false;
    //     valArray[i] = 0;
    // }

    for (Idx r = 0; r < m; ++r)
    {
        for (Size_t off = a.rowBeginOffset[r]; off < a.rowBeginOffset[r + 1]; ++off)
        {
            Idx a_col = a.colIdx[off];
            Val a_val = a.values[off];
            for (Size_t offB = b.rowBeginOffset[a_col]; offB < b.rowBeginOffset[a_col + 1]; ++offB)
            {
                Idx b_col = b.colIdx[offB];
                Val b_val = b.values[offB];

                // if (!bitmat[b_col])
                //     nonZeroCol[rowNnz++] = b_col;
                // bitmat[b_col] = true;
                // valArray[b_col] += a_val * b_val;
                themap[b_col] += a_val * b_val;
            }
        }
        NNZIdx rowNnz = 0;
        for(auto s : themap){
            sortBuffer[rowNnz].first = s.first;
            sortBuffer[rowNnz].second = s.second;
            rowNnz++;
        }
        if constexpr (method == SpMMDenseSort::row_quicksort){
            std::sort(sortBuffer.get(),sortBuffer.get()+rowNnz,[](std::pair<Idx,Val> a, std::pair<Idx,Val> b){
                return a.first < b.first;
            });
        }
        themap.clear();

        c.rowBeginOffset[r] = matrixNnz;
        for (NNZIdx i = 0; i < rowNnz; ++i)
        {
            Size_t pos = matrixNnz + i;
            c.colIdx[pos] = sortBuffer[i].first;
            c.values[pos] = sortBuffer[i].second;
        }
        matrixNnz += rowNnz;
        rowNnz = 0;
    }

    c.nnz = matrixNnz;
    c.rowBeginOffset.back() = matrixNnz;

    if constexpr (method == SpMMDenseSort::row_quicksort 
            || method == SpMMDenseSort::no_sort){
        timmerHelper.end();
        c.colIdx.resize(matrixNnz);
        c.colIdx.shrink_to_fit();
        c.values.resize(matrixNnz);
        c.values.shrink_to_fit();
        return c;
    } else {
        // method == SpMMDenseSort::mat_linklist
        auto temp =  sortAllRow(c);
        timmerHelper.end();
        return temp;
    }
}