#include <fstream>
#include <iostream>
#include <string>
#include "SparseMatTool/serialization.hpp"



std::ofstream &operator<<(std::ofstream &of, const CSR &csr)
{
	// of << csr.m << csr.n << csr.nnz;
	writeRaw(of,csr.m);
	writeRaw(of,csr.n);
	writeRaw(of,csr.nnz);
	WriteBufferBinary(of, csr.rowBeginOffset);
	WriteBufferBinary(of, csr.colIdx);
	WriteBufferBinary(of, csr.values);
	return of;
}

std::ifstream &operator>>(std::ifstream &infile, CSR &csr)
{
	// infile >> csr.m >> csr.n >> csr.nnz;
	readRaw(infile,csr.m);
	readRaw(infile,csr.n);
	readRaw(infile,csr.nnz);
	ReadBufferBinary(infile, csr.rowBeginOffset);
	ReadBufferBinary(infile, csr.colIdx);
	ReadBufferBinary(infile, csr.values);
	return infile;
}

std::ostream &operator<<(std::ostream &os, const CSR &csr)
{
	os << "m is " << csr.m << std::endl;
	os << "n is " << csr.n << std::endl;
	os << "nnz is " << csr.nnz << std::endl;
	for (Idx row = 0; row < csr.m; ++row)
	{
		Size_t rowStart = csr.rowBeginOffset[row];
		Size_t rowEnd = csr.rowBeginOffset[row + 1];
		for (Size_t off = rowStart; off < rowEnd; ++off)
		{
			os << "[" << off << "]\t" << row << '\t' << csr.colIdx[off] << '\t' << csr.values[off] << '\t' << std::endl;
		}
	}
	return os;
}

std::ostream &operator<<(std::ostream &os, const COO &coo)
{
	std::cout << "m is " << coo.m << std::endl;
	std::cout << "n is " << coo.n << std::endl;
	std::cout << "nnz is " << coo.nnz << std::endl;
	for (Size_t i = 0; i < coo.nnz; ++i)
	{
		std::cout << "[" << i << "]\t" << coo.xidxs[i] << '\t' << coo.yidxs[i] << '\t' << coo.values[i] << '\t' << std::endl;
	}
	return os;
}

void printMatrixBrief(std::ostream &os, const CSR &csr, std::string name)
{
	printf("Matrix [%s]: \tm=%7lu, \t n=%7lu, \t nnz = %7lu, \t edgeF = %7.2f\n", name.c_str(), csr.m, csr.n, csr.nnz,double(csr.nnz)/csr.m);
}

CSR loadCSR(const std::string &fileName)
{
	CSR csr;
	std::ifstream infile;
	infile.exceptions(std::ios::failbit);
	infile.open(fileName, std::ifstream::in | std::ifstream::binary);
	infile >> csr;
	infile.close();
	return csr;
}

void writeCSR(const std::string &fileName, const CSR &csr)
{
	std::ofstream of;
	of.exceptions(std::ios::failbit);
	of.open(fileName, std::ofstream::out | std::ofstream::binary);
	of << csr;
	of.close();
}