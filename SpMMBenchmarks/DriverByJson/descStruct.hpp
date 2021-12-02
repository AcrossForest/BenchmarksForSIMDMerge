#pragma once
#include <string>
#include <map>
#include "SparseMatTool/format.hpp"
#include "Benchmarking/benchmarking.hpp"

struct MatrixDesc
{
    std::string fileName;  // such as ./data/1000-1000-10000.bincsr
    Idx m, n;
    Size_t nnz;
    double edgeFactor; // Equals nnz/m
};

struct WorkLoadDescription
{
    std::map<std::string,MatrixDesc> inputs;
    std::map<std::string,MatrixDesc> outputs;
    std::string kernelName;
};

struct ExecuteReport : public WorkLoadDescription
{
    Timmer timmer;
    ExecuteReport() = default;
    ExecuteReport(const WorkLoadDescription& ref):WorkLoadDescription(ref){}
};
