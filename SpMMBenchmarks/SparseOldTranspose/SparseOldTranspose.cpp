#include <algorithm>
#include "SparseMatTool/format.hpp"
#include "SparseOldTranspose/SparseOldTranspose.hpp"

CSR oldTranspose(const CSR &a)
{
    const Size_t sentinalVal = std::numeric_limits<Size_t>::max();
    Buffer<Size_t> lastSeenPosition(a.n, sentinalVal);
    Buffer<Size_t> nextSameCol(a.nnz);
    Buffer<Idx> rowNum(a.nnz);

    // Can't use row>=0 due to unsigned number underflow
    for (Idx row = a.m; row > 0;)
    {
        --row;
        Size_t rowBegin = a.rowBeginOffset[row];
        Size_t rowEnd = a.rowBeginOffset[row + 1];
        for (Size_t col_c_off = rowBegin; col_c_off < rowEnd; ++col_c_off)
        {
            Idx col = a.colIdx[col_c_off];
            rowNum[col_c_off] = row;
            nextSameCol[col_c_off] = lastSeenPosition[col];
            lastSeenPosition[col] = col_c_off;
        }
    }

    CSR b;
    b.m = a.n;
    b.n = a.m;
    b.nnz = a.nnz;
    b.values.resize(a.nnz);
    b.colIdx.resize(a.nnz);
    b.rowBeginOffset.resize(a.n + 1);

    Size_t b_nnz_fill = 0;

    for (Idx a_col = 0; a_col < a.n; ++a_col)
    {
        Idx b_row = a_col;
        b.rowBeginOffset[b_row] = b_nnz_fill;

        Size_t a_col_c_off = lastSeenPosition[a_col];
        while (a_col_c_off != sentinalVal)
        {
            // Element in a:
            // Offset: a_col_c_off (existing)
            // Row: a_row
            Idx a_row = rowNum[a_col_c_off];
            // Col: a_col (existing)
            Val a_val = a.values[a_col_c_off];

            // Now push to b
            // Row: b_row = a_col
            // Col: b_col = a_row
            b.colIdx[b_nnz_fill] = a_row;
            b.values[b_nnz_fill] = a_val;
            b_nnz_fill++;

            a_col_c_off = nextSameCol[a_col_c_off];
        }
    }

    b.rowBeginOffset.back() = a.nnz;
    return b;
}

CSR sortAllRow(const CSR &a)
{
    const Size_t sentinalVal = std::numeric_limits<Size_t>::max();
    Buffer<Size_t> linkListHead(a.n, sentinalVal);
    Buffer<Size_t> linkListPointers(a.nnz);
    Buffer<Idx> rowLookup(a.nnz);

    for (Idx row = a.m; row > 0;)
    {
        --row;
        Size_t rowBegin = a.rowBeginOffset[row];
        Size_t rowEnd = a.rowBeginOffset[row + 1];

        for (Size_t col_c_off = rowBegin; col_c_off < rowEnd; ++col_c_off)
        {
            Idx col = a.colIdx[col_c_off];
            linkListPointers[col_c_off] = linkListHead[col];
            linkListHead[col] = col_c_off;
            rowLookup[col_c_off] = row;
        }
    }

    CSR b;
    b.m = a.m;
    b.n = a.n;
    b.nnz = a.nnz;
    b.rowBeginOffset = a.rowBeginOffset;
    b.colIdx.resize(a.colIdx.size());
    b.values.resize(a.values.size());

    Buffer<Size_t> eachRowFillOffset = a.rowBeginOffset;

    for (Idx col = 0; col < a.n; ++col)
    {
        Size_t a_col_c_off = linkListHead[col];
        while (a_col_c_off != sentinalVal)
        {
            Idx _row = rowLookup[a_col_c_off];
            Idx _col = col;
            Val _val = a.values[a_col_c_off];

            Size_t b_col_c_off = eachRowFillOffset[_row];
            b.colIdx[b_col_c_off] = _col;
            b.values[b_col_c_off] = _val;
            eachRowFillOffset[_row]++;
            a_col_c_off = linkListPointers[a_col_c_off];
        }
    }

    return b;
}

CSR sortAllRowDebug(const CSR &a)
{
    Buffer<Size_t> arr_col_c_0(a.n);
    CSR b(a);
    for (Idx row = 0; row < a.m; ++row)
    {
        Size_t rowBegin = a.rowBeginOffset[row];
        Size_t rowEnd = a.rowBeginOffset[row + 1];

        NNZIdx thisRowNNZ = rowEnd - rowBegin;
        for (NNZIdx col_c_0 = 0; col_c_0 < thisRowNNZ; ++col_c_0)
        {
            arr_col_c_0[col_c_0] = col_c_0;
        }
        std::sort(arr_col_c_0.begin(),
                  arr_col_c_0.begin() + thisRowNNZ,
                  [rowBegin, &a](NNZIdx x1, NNZIdx x2) {
                      return a.colIdx[rowBegin + x1] < a.colIdx[rowBegin + x2];
                  });
        
        for (NNZIdx col_c_0 = 0; col_c_0 < thisRowNNZ; ++col_c_0)
        {
            b.colIdx[rowBegin + col_c_0] = a.colIdx[rowBegin + arr_col_c_0[col_c_0]];
            b.values[rowBegin + col_c_0] = a.values[rowBegin + arr_col_c_0[col_c_0]];
        }
    }
    return b;
}