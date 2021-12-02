#include <string>
#include <exception>
#include <ranges>
#include <cmath>
#include <fstream>
#include "nlohmann/json.hpp"
#include "SparseMatTool/format.hpp"
#include "Benchmarking/benchmarking.hpp"
#include "DriverByJson/DriverByJson.hpp"

template <class T>
struct Tag
{
};

using json = nlohmann::ordered_json;
json dump(const MatrixDesc &matdesc)
{
    json j = {
        {"fileName", matdesc.fileName},
        {"m", matdesc.m},
        {"n", matdesc.n},
        {"nnz", matdesc.nnz},
        {"edgeFactor", matdesc.edgeFactor}};
    return j;
}

MatrixDesc extract(Tag<MatrixDesc>, const json &j, bool readMeta)
{
    MatrixDesc m;
    m.fileName = j["fileName"];
    if (readMeta)
    {
        m.m = j["m"];
        m.n = j["n"];
        m.nnz = j["nnz"];
        m.edgeFactor = j["edgeFactor"];
        if (std::abs(m.nnz / m.m - m.edgeFactor) > 0.5)
        {
            throw std::runtime_error("Matrix description's edgeFactor is not consistent.");
        }
    }
    return m;
}

json dump(const Timmer &timmer)
{
    json j;
    for (const auto &recordPair : timmer.records)
    {
        const auto name = recordPair.name;
        const auto vec = *recordPair.data;
        auto statResult = statAnalysis(vec.begin(), vec.end());

        j[name] = {{"operation", name},
                   {"mean", statResult.mean},
                   {"sd", statResult.sd},
                   {"nSamples", statResult.n},
                   {"records", vec}};
    }
    return j;
}

WorkLoadDescription extract(Tag<WorkLoadDescription>, const json &j)
{
    WorkLoadDescription work;
    for (auto &[paramName, elem] : j["inputs"].items())
    {
        work.inputs.emplace(paramName, extract(Tag<MatrixDesc>(), elem, true));
    }
    for (auto &[paramName, elem] : j["outputs"].items())
    {
        work.outputs.emplace(paramName, extract(Tag<MatrixDesc>(), elem, false));
    }

    work.kernelName = j["kernelName"];
    return work;
}

json dump(const ExecuteReport &exec)
{
    auto dumpMatrixDict = [](const std::map<std::string, MatrixDesc> &vecDesc) -> json {
        json j;
        for (const auto &[paramName, matDesc] : vecDesc)
        {
            j[paramName] = dump(matDesc);
        }
        return j;
    };

    json j = {
        {"kernelName", exec.kernelName},
        {"inputs", dumpMatrixDict(exec.inputs)},
        {"outputs", dumpMatrixDict(exec.outputs)},
        {"timmer", dump(exec.timmer)}};
    return j;
}
std::string readString(std::string fileName)
{
    // https://stackoverflow.com/a/525103
    std::ifstream in;
    // Must open in binary mode, otherwise tellg() didn't tell the correst 
    // filesize in windows system.
    in.open(fileName, std::ios::ate | std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file " + fileName);
    }
    auto fileSize = in.tellg();
    in.seekg(0, std::ios::beg);

    std::vector<char> bytes(fileSize);
    in.read(bytes.data(), fileSize);
    in.close();
    return std::string(bytes.data(), fileSize);
}

WorkLoadDescription loadWorkloadDesc(std::string workLoadDescFile)
{
    std::string content = readString(workLoadDescFile);
    return extract(Tag<WorkLoadDescription>(), json::parse(content));
}

void WriteExecuteReport(const ExecuteReport &exec, std::string execReportFile)
{
    json j = dump(exec);
    std::ofstream of;
    of.exceptions(std::ios::failbit);
    of.open(execReportFile);
    of << j.dump(4);
    of.close();
}