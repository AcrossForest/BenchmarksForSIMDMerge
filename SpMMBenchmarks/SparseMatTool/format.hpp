#pragma once
#include <stdint.h>
#include <tuple>
#include <vector>
#include "specialAllocator.hpp"

template <class T>
using Buffer = std::vector<T,default_init_allocator<T>>;

using Size_t = std::size_t;
using Idx = uint32_t;
using NNZIdx = uint32_t;
using Val = float;

struct CSR
{
	Size_t m = 0, n = 0, nnz = 0;
	Buffer<Size_t> rowBeginOffset;
	Buffer<Idx> colIdx;
	Buffer<Val> values;

	 // Not compile for unkown reson.
	// auto operator<=>(const CSR&) const = default;
	bool operator==(const CSR &other) const
	{
		return m == other.m &&
			   n == other.n &&
			   nnz == other.nnz &&
			   colIdx == other.colIdx &&
			   rowBeginOffset == other.rowBeginOffset &&
			   values == other.values;
	}

	bool almostEqual(const CSR &other, double epsilon = 1e-3)
	{
		bool equal = m == other.m &&
					 n == other.n &&
					 nnz == other.nnz &&
					 colIdx == other.colIdx &&
					 rowBeginOffset == other.rowBeginOffset;
		for (Size_t i = 0; i < values.size(); ++i)
		{
			equal &= (values[i] - other.values[i]) < epsilon &&
					 (other.values[i] - values[i]) < epsilon;
		}
		return equal;
	}
};

struct COO
{
	enum class Status
	{
		construct,
		sorted
	} status = Status::construct;
	Size_t nnz = 0;
	Buffer<Idx> xidxs, yidxs;
	Buffer<Val> values;

	// Available only when it is in sorted status.
	Idx m = 0, n = 0;

	void reserve(Size_t possibleNnz)
	{
		xidxs.reserve(possibleNnz);
		yidxs.reserve(possibleNnz);
		values.reserve(possibleNnz);
	}

	void insert(Idx xidx, Idx yidx, Val val)
	{
		status = Status::construct;
		xidxs.push_back(xidx);
		yidxs.push_back(yidx);
		values.push_back(val);
		++nnz;
	}
	void sort();
};

CSR COO2CSR(COO &coo);
