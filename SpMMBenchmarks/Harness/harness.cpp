#include <iostream>
#include <exception>
#include <functional>
#include "DriverByJson/descStruct.hpp"
#include "DriverByJson/DriverByJson.hpp"
#include "Harness/harness.hpp"
#include "SparseMMVecAccum/SpMMVecAccum.hpp"
#include "SparseMMHeapAccum/SparseMMHeapAccum.hpp"
#include "SparseMMTimOptimize/sparseGemm.hpp"
#include "SparseMMTimSortAlike/SparseMMTimSortAlike.hpp"
#ifdef __USE_YUSUKE__
#include "ExternalBenchYusukeNagasaka/YusukeNagasaka.hpp"
#endif

#ifdef __USE_EIGEN_BENCH__
#include "ExternEigen.hpp"
#endif

int main(int argv, char **argc)
{
    std::cout << "Program starts." << std::endl;
    
    if (argv != 5)
    {
        printf("usage: %s path/to/the/workloadDesc.json path/to/the/outputReport.json scalar_repeat simd_repeat\n", argc[0]);
        return 0;
    }
    std::cout << "CMD:\t";
    for(int i=0; i<argv; ++i){
        std::cout << argc[i] << "\t";
    }
    std::cout << std::endl;

    std::cout << "Now I am here." << std::endl;
    int scalar_repeat = std::stoi(argc[3]);
    int simd_repeat = std::stoi(argc[4]);
    // printf("Now I am here.\n");
    std::string workloadDescFile(argc[1]);
    std::cout << "Now I am second here." << std::endl;
    WorkLoadDescription workloadDesc = loadWorkloadDesc(workloadDescFile);

    SpGemmHarness harness(workloadDesc);

    ExecuteReport exec;

    if (workloadDesc.kernelName == "SparseMMVecAccum_quickSort" || workloadDesc.kernelName == "Dense")
    {
        exec = harness.bench(scalar_repeat,[](TimmerHelper& timmerHelper, const CSR &A, const CSR &B) {
            return SpMM_DenseVec(
                timmerHelper,
                SpMMSortTag<SpMMDenseSort::row_quicksort>{},
                A, B);
        });
    }
    else if (workloadDesc.kernelName == "SparseMMVecAccum_noSort")
    {
        exec = harness.bench(scalar_repeat,[](TimmerHelper& timmerHelper, const CSR &A, const CSR &B) {
            return SpMM_DenseVec(
                timmerHelper,
                SpMMSortTag<SpMMDenseSort::no_sort>{},
                A, B);
        });
    }
    else if (workloadDesc.kernelName == "SparseMMVecAccum_sortMat")
    {
        exec = harness.bench(scalar_repeat,[](TimmerHelper& timmerHelper, const CSR &A, const CSR &B) {
            return SpMM_DenseVec(
                timmerHelper,
                SpMMSortTag<SpMMDenseSort::mat_linklist>{},
                A, B);
        });
    }
    else if (workloadDesc.kernelName == "SparseMMVecHash" || workloadDesc.kernelName == "Hash")
    {
        exec = harness.bench(scalar_repeat,[](TimmerHelper& timmerHelper, const CSR &A, const CSR &B) {
            return SpMM_Hash(
                timmerHelper,
                SpMMSortTag<SpMMDenseSort::row_quicksort>{},
                A, B);
        });
    }
    else if (workloadDesc.kernelName == "SparseMMHeapAccum" || workloadDesc.kernelName == "Heap")
    {
        exec = harness.bench(scalar_repeat,[](TimmerHelper& timmerHelper, const CSR &A, const CSR &B) {
            return SpMM_Heap(timmerHelper,A, B);
        });
    }
    else if (workloadDesc.kernelName == "SparseMMTimSortAlike" || workloadDesc.kernelName == "Merge")
    {
        exec = harness.bench(scalar_repeat,[](TimmerHelper& timmerHelper, const CSR &A, const CSR &B) {
            return SpMM_TimSortAlike(timmerHelper,directVectorAdd,A, B);
        });
    }
    else if (workloadDesc.kernelName == "SparseMMTimSortAlikeSpSp" || workloadDesc.kernelName == "ProposedSIMD")
    {
        exec = harness.bench(simd_repeat,[](TimmerHelper& timmerHelper, const CSR &A, const CSR &B) {
            return SpMM_TimSortAlike(timmerHelper,instBasedVectorAdd,A, B);
        });
    }
    else if (workloadDesc.kernelName == "SparseMMTimOptimized")
    {
        using namespace OptimizedTim;
        exec = harness.bench(scalar_repeat,[](TimmerHelper& timmerHelper, const CSR &A, const CSR &B) {
            return SpMM_TimSortOptimized(timmerHelper,ScalarToolkit{},A, B);
        });
    }
    else if (workloadDesc.kernelName == "SparseMMTimOptimizedSpSp")
    {
        using namespace OptimizedTim;
        exec = harness.bench(simd_repeat,[](TimmerHelper& timmerHelper, const CSR &A, const CSR &B) {
            return SpMM_TimSortOptimized(timmerHelper,SIMDToolkit{},A, B);
        });
    }
#ifdef __USE_EIGEN_BENCH__
    else if (workloadDesc.kernelName == "SparseMMEigen" || workloadDesc.kernelName == "Eigen")
    {
        // exec = harness.bench([](TimmerHelper& timmerHelper, const CSR &A, const CSR &B) {
        //     return SpMM_TimSortAlike(timmerHelper,instBasedVectorAdd,A, B);
        // });
        exec = harness.bench(scalar_repeat,
            CSR2EigenCSR,
            [](const EigenCSR& a, const EigenCSR& b) {
                return a*b;
            },
            EigenCSR2CSR
            );
    }
#endif
#ifdef __USE_YUSUKE__
    else if (workloadDesc.kernelName == "SparseMMExternalYuHeap")
    {
        using namespace externalBench;
        using Yu32 = Yu<Idx,Val>;
        exec = harness.bench(scalar_repeat,
            Yu32::fromCSR2YuCSC_Transpose,
            [](const Yu32::YuCSC &yu_a, const Yu32::YuCSC &yu_b) {
                Yu32::YuCSC temp;
                HeapSpGEMM(yu_a, yu_b, temp,
                           std::multiplies<Val>{},
                           std::plus<Val>{});
                return temp;
            },
            Yu32::fromYuCSC2CSR_Transpose
            );
    }
    else if (workloadDesc.kernelName == "SparseMMExternalYuHash")
    {
        using namespace externalBench;
        using Yu32 = Yu<Idx,Val>;
        exec = harness.bench(scalar_repeat,
            Yu32::fromCSR2YuCSR,
            [](const Yu32::YuCSR &yu_a, const Yu32::YuCSR &yu_b) {
                Yu32::YuCSR temp;
                HashSpGEMM<false,true>(yu_a, yu_b, temp,
                           std::multiplies<Val>{},
                           std::plus<Val>{});
                return temp;
            },
            Yu32::fromYuCSR2CSR
            );
    }
    else if (workloadDesc.kernelName == "SparseMMExternalYuHash-NoSort")
    {
        using namespace externalBench;
        using Yu32 = Yu<Idx,Val>;
        exec = harness.bench(scalar_repeat,
            Yu32::fromCSR2YuCSR,
            [](const Yu32::YuCSR &yu_a, const Yu32::YuCSR &yu_b) {
                Yu32::YuCSR temp;
                HashSpGEMM<false,false>(yu_a, yu_b, temp,
                           std::multiplies<Val>{},
                           std::plus<Val>{});
                return temp;
            },
            Yu32::fromYuCSR2CSR
            );
    }
    else if (workloadDesc.kernelName == "SparseMMExternalYuHashVec")
    {
        using namespace externalBench;
        using Yu64 = Yu<long long int,Val>;
        exec = harness.bench(simd_repeat,
            Yu64::fromCSR2YuCSR,
            [](const Yu64::YuCSR &yu_a, const Yu64::YuCSR &yu_b) {
                Yu64::YuCSR temp;
                HashSpGEMM<true,true>(yu_a, yu_b, temp,
                           std::multiplies<Val>{},
                           std::plus<Val>{});
                return temp;
            },
            Yu64::fromYuCSR2CSR
            );
    }
    else if (workloadDesc.kernelName == "SparseMMExternalYuHashVec-NoSort")
    {
        using namespace externalBench;
        using Yu64 = Yu<long long int,Val>;
        exec = harness.bench(simd_repeat,
            Yu64::fromCSR2YuCSR,
            [](const Yu64::YuCSR &yu_a, const Yu64::YuCSR &yu_b) {
                Yu64::YuCSR temp;
                HashSpGEMM<true,false>(yu_a, yu_b, temp,
                           std::multiplies<Val>{},
                           std::plus<Val>{});
                return temp;
            },
            Yu64::fromYuCSR2CSR
            );
    }
#endif
    else
    {
        throw std::runtime_error("Unkown kernel:" + workloadDesc.kernelName);
    }

    std::string reportFile(argc[2]);
    WriteExecuteReport(exec, reportFile);
    printf("I finished normallly.\n");
    std::cout << argc[2] << std::endl;
}