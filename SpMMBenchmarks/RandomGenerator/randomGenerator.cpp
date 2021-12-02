#include <iostream>
#include <unordered_set>
#include <random>
#include "SparseMatTool/format.hpp"
#include "SparseMatTool/serialization.hpp"
#include <fstream>
#include <algorithm>

COO createRandomMatrx(Idx m, Idx n, Size_t nnz,Idx uniqueCols = 0)
{
	if (uniqueCols == 0) uniqueCols = n;
	nnz = std::min<Size_t>(m*n,nnz);
	std::ranlux48_base gen(741);
	std::uniform_int_distribution<Idx> dism(0, m - 1), disn(0, n - 1);

	auto hashFun = [n](std::pair<Idx, Idx> a) -> Size_t {
		return a.first + a.second * n;
	};
	std::unordered_set<std::pair<Idx, Idx>, decltype(hashFun)> hashTable(0, hashFun);
	hashTable.reserve(nnz);

	int attempts = 0;
	while (hashTable.size() < nnz && attempts < 2*nnz)
	{
		hashTable.insert({dism(gen), (n/uniqueCols) * (disn(gen)%uniqueCols) });
	}
	COO coo;
	coo.m = m;
	coo.n = n;
	coo.nnz = 0;
	coo.reserve(nnz);

	std::uniform_real_distribution<Val> disv(0, 1);
	for (auto p : hashTable)
	{
		coo.insert(p.first, p.second, disv(gen));
	}

	coo.sort();
	return coo;
}

int main(int argv, char **argc)
{
	if (argv != 6)
	{
		printf("Usage: %s m n nnz uniqueCol outFileName\n", argc[0]);
		return -1;
	}
	int m = std::stoi(argc[1]);
	int n = std::stoi(argc[2]);
	int nnz = std::stoi(argc[3]);
	int uniqueCol = std::stoi(argc[4]);

	auto coo = createRandomMatrx(m, n, nnz, uniqueCol);
	auto csr = COO2CSR(coo);

	writeCSR(argc[5],csr);

}