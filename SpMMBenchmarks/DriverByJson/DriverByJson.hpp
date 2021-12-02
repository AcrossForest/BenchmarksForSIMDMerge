#pragma once
#include <string>
#include "DriverByJson/descStruct.hpp"

std::string readString(std::string fileName);

WorkLoadDescription loadWorkloadDesc(std::string workLoadDescFile);

void WriteExecuteReport(const ExecuteReport &exec, std::string execReportFile);