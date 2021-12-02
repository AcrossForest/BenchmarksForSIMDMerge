#include <fstream>
#include <exception>
#include "DriverByJson/safeIO.hpp"
#include "DriverByJson/descStruct.hpp"
#include "SparseMatTool/format.hpp"
#include "SparseMatTool/serialization.hpp"



CSR safeLoadCSR(const MatrixDesc& desc){
    std::ifstream infile;
    infile.exceptions(std::ios::failbit);
    infile.open(desc.fileName,std::ios::binary);

    CSR csr;
    readRaw(infile,csr.m);
	readRaw(infile,csr.n);
	readRaw(infile,csr.nnz);

    if(desc.m != csr.m || desc.n != csr.n || desc.nnz != csr.nnz){
        throw std::runtime_error("The matrix loaded in file " 
                + desc.fileName 
                + " is not the same as desc.\n"
                + "(m,n,nnz) in binary file is (" 
                + std::to_string(csr.m) + "," 
                + std::to_string(csr.n) + "," 
                + std::to_string(csr.nnz) + "," 
                + ") "
                 );
    }

	ReadBufferBinary(infile, csr.rowBeginOffset);
	ReadBufferBinary(infile, csr.colIdx);
	ReadBufferBinary(infile, csr.values);
    infile.close();
    return csr;
}

MatrixDesc safeWriteCSR(const MatrixDesc& oldDesc,const CSR& csr){
    MatrixDesc newDesc(oldDesc);
    newDesc.m = csr.m;
    newDesc.n = csr.n;
    newDesc.nnz = csr.nnz;
    newDesc.edgeFactor = double(csr.nnz) / csr.m;

    std::ofstream of;
    of.exceptions(std::ios::failbit);
    of.open(newDesc.fileName,std::ios::binary);
    writeRaw(of,csr.m);
	writeRaw(of,csr.n);
	writeRaw(of,csr.nnz);
	WriteBufferBinary(of, csr.rowBeginOffset);
	WriteBufferBinary(of, csr.colIdx);
	WriteBufferBinary(of, csr.values);
    of.close();
    return newDesc;
}