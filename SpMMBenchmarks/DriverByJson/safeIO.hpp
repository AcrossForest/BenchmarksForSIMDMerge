#pragma once
#include <fstream>
#include <exception>
#include "DriverByJson/descStruct.hpp"
#include "SparseMatTool/format.hpp"

CSR safeLoadCSR(const MatrixDesc& desc);
MatrixDesc safeWriteCSR(const MatrixDesc& oldDesc,const CSR& csr);