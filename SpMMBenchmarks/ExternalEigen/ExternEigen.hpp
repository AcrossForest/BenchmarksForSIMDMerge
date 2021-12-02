#include <Eigen/SparseCore>
#include "SparseMatTool/format.hpp"
#include <memory>
#include <tuple>
using EigenCSR = Eigen::SparseMatrix<float,Eigen::RowMajor,int>;

EigenCSR CSR2EigenCSR(const CSR& csr){
    EigenCSR a(csr.m,csr.n);
    auto buffer = std::make_unique<Eigen::Triplet<Val,Idx>[]>(csr.nnz);
    for(size_t r=0; r<csr.m; ++r){
        for(size_t off=csr.rowBeginOffset[r];off<csr.rowBeginOffset[r+1];++off){
            Idx col = csr.colIdx[off];
            Val val = csr.values[off];
            buffer[off] = {r,col,val};
        }
    }
    a.setFromTriplets(buffer.get(),buffer.get()+csr.nnz);
    return a;
}

CSR EigenCSR2CSR(EigenCSR& eigenCSR){
    eigenCSR.makeCompressed();
    CSR csr;
    csr.nnz = eigenCSR.nonZeros();
    csr.m = eigenCSR.rows();
    csr.n = eigenCSR.cols();

    csr.rowBeginOffset.resize(csr.m+1);
    std::copy(eigenCSR.outerIndexPtr(),eigenCSR.outerIndexPtr()+csr.m,csr.rowBeginOffset.begin());
    csr.rowBeginOffset.back() = csr.nnz;

    csr.colIdx.resize(csr.nnz);
    std::copy(eigenCSR.innerIndexPtr(),eigenCSR.innerIndexPtr()+csr.nnz,csr.colIdx.begin());

    csr.values.resize(csr.nnz);
    std::copy(eigenCSR.valuePtr(),eigenCSR.valuePtr()+eigenCSR.nonZeros(),csr.values.begin());
    return csr;
}