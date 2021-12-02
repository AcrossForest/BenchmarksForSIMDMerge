#pragma once
#include <fstream>
#include <string>
#include <type_traits>
#include "SparseMatTool/format.hpp"

template<class T>
std::enable_if_t<std::is_trivial_v<T>>
readRaw(std::ifstream &infile,T& data){
	infile.read(reinterpret_cast<char *>(&data),sizeof(T));
}

template<class T>
std::enable_if_t<std::is_trivial_v<T>>
writeRaw(std::ofstream &of, const T&data){
	of.write(reinterpret_cast<const char*>(&data),sizeof(T));
}


template <class ElemType>
std::enable_if_t<std::is_trivial_v<ElemType>>
WriteBufferBinary(std::ofstream& of, const Buffer<ElemType>& buff)
{
	Size_t length = buff.size();
	writeRaw(of,length);
	if (length)
	{
		const auto start = reinterpret_cast<const char*>(&buff[0]);
		of.write(start, length * sizeof(ElemType));
	}
}

template <class ElemType>
std::enable_if_t<std::is_trivial_v<ElemType>>
ReadBufferBinary(std::ifstream& infile, Buffer<ElemType>& buff)
{
	Size_t length;
	readRaw(infile,length);
	if (length)
	{
		buff.resize(length);
		auto start = reinterpret_cast<char*>(&buff[0]);
		infile.read(start, length * sizeof(ElemType));
	}
}

std::ofstream& operator<<(std::ofstream& of, const CSR& csr);

std::ifstream& operator>>(std::ifstream& infile, CSR& csr);

void printMatrixBrief(std::ostream& os, const CSR& csr,std::string name);

std::ostream& operator<<(std::ostream& os, const CSR& csr);

std::ostream& operator<<(std::ostream& os, const COO&coo);

CSR loadCSR(const std::string& fileName);

void writeCSR(const std::string& fileName, const CSR& csr);