#include <algorithm>
#include "YusukeNagasaka/CSC.h"
#include "YusukeNagasaka/CSR.h"
#include "YusukeNagasaka/heap_mult.h"
#include "YusukeNagasaka/hash_mult.h"
#include "SparseMatTool/format.hpp"

namespace externalBench
{
    template <class YuIdx, class YuVal>
    struct Yu
    {
        using YuCSR = YusukeNagasaka::CSR<YuIdx, YuVal>;
        using YuCSC = YusukeNagasaka::CSC<YuIdx, YuVal>;

        static YuCSR fromCSR2YuCSR(const CSR &a)
        {
            YuCSR yucsr(a.nnz, a.m, a.n);

            std::copy(a.colIdx.data(), a.colIdx.data() + a.nnz, yucsr.colids);
            std::copy(a.values.data(), a.values.data() + a.nnz, yucsr.values);
            std::copy(a.rowBeginOffset.data(),
                      a.rowBeginOffset.data() + a.m + 1,
                      yucsr.rowptr);
            return yucsr;
        }

        static CSR fromYuCSR2CSR(const YuCSR &yucsr)
        {
            CSR a;
            a.m = yucsr.rows;
            a.n = yucsr.cols;
            a.nnz = yucsr.nnz;
            a.rowBeginOffset.assign(yucsr.rowptr, yucsr.rowptr + yucsr.rows + 1);
            a.colIdx.assign(yucsr.colids, yucsr.colids + yucsr.nnz);
            a.values.assign(yucsr.values, yucsr.values + yucsr.nnz);
            return a;
        }

        static YuCSC fromCSR2YuCSC_Transpose(const CSR &a)
        {
            YuCSC yucsc(a.nnz, a.n, a.m, 0);

            std::copy(a.rowBeginOffset.data(),
                      a.rowBeginOffset.data() + a.m + 1,
                      yucsc.colptr);
            std::copy(a.colIdx.data(),
                      a.colIdx.data() + a.nnz,
                      yucsc.rowids);
            std::copy(a.values.data(),
                      a.values.data() + a.nnz,
                      yucsc.values);
            return yucsc;
        }

        static CSR fromYuCSC2CSR_Transpose(const YuCSC &yucsc)
        {
            CSR a;
            a.nnz = yucsc.nnz;
            a.m = yucsc.cols;
            a.n = yucsc.rows;
            a.rowBeginOffset.assign(yucsc.colptr, yucsc.colptr + yucsc.cols + 1);
            a.colIdx.assign(yucsc.rowids, yucsc.rowids + yucsc.nnz);
            a.values.assign(yucsc.values, yucsc.values + yucsc.nnz);
            return a;
        }

    };
    using YusukeNagasaka::HashSpGEMM;
    using YusukeNagasaka::HeapSpGEMM;

} // namespace externalBench