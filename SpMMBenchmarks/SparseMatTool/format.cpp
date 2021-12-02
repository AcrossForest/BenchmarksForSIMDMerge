#include <algorithm>
#include <exception>
#include <stdexcept>
#include "SparseMatTool/format.hpp"

void COO::sort()
{
    Buffer<Size_t> buf(nnz);
    for (Size_t i = 0; i < nnz; ++i)
    {
        buf[i] = i;
    }
    std::sort(buf.begin(), buf.end(),
              [this](Size_t a, Size_t b) {
                  return std::make_pair(xidxs[a], yidxs[a]) < std::make_pair(xidxs[b], yidxs[b]);
              });
    Buffer<Idx> nxidx(nnz), nyidx(nnz);
    Buffer<Val> nvalues(nnz);

    for (Size_t i = 0; i < nnz; ++i)
    {
        Size_t from = buf[i];
        nxidx[i] = xidxs[from];
        nyidx[i] = yidxs[from];
        nvalues[i] = values[from];
    }

    xidxs.swap(nxidx);
    yidxs.swap(nyidx);
    values.swap(values);
    status = Status::sorted;
}

CSR COO2CSR(COO &coo)
{
    if (coo.status != COO::Status::sorted)
        coo.sort();
    CSR csr;
    csr.m = coo.m;
    csr.n = coo.n;
    csr.nnz = coo.nnz;
    csr.rowBeginOffset.resize(csr.m + 1);
    csr.colIdx.assign(coo.yidxs.begin(), coo.yidxs.end());
    csr.values.assign(coo.values.begin(), coo.values.end());

    // Find the start index of each row
    // 0 0 0 1 1 1 1 2 2 3 3 3
    // 0     1       2   3     4
    // Case 1: 0
    // Case 2: where xidx[i] != xidx[i-1]
    // Case 3: the index of last element + 1

    Size_t scan = 0;
    for (Idx row = 0; row < coo.m; ++row)
    {
        csr.rowBeginOffset[row] = scan;
        Size_t testInit = 0;
        while (scan < coo.nnz && coo.xidxs[scan] == row){
            if(testInit > coo.yidxs[scan]){
                throw std::logic_error("Should be monotone \n");
            }
            testInit = coo.yidxs[scan];
            scan++;
        }
    }
    csr.rowBeginOffset.back() = coo.nnz;
    return csr;
}
