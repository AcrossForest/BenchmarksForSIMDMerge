#pragma once
#include <string>
#include "Benchmarking/benchmarking.hpp"
#include "SparseMatTool/format.hpp"
#include "SparseMatTool/serialization.hpp"
#include "DriverByJson/descStruct.hpp"
#include "DriverByJson/safeIO.hpp"
#include "M5MagicInst/m5ops.h"

struct SpGemmHarness{
    WorkLoadDescription workloadDesc;
    MatrixDesc descA,descB,descC;

    SpGemmHarness(const WorkLoadDescription& workloadDesc):workloadDesc(workloadDesc){
        descA = workloadDesc.inputs.at("matrixA");
        descB = workloadDesc.inputs.at("matrixB");
        descC = workloadDesc.outputs.at("matrixC");
    }

    template <class Callable>
    ExecuteReport bench(int repeat, Callable fun){
        ExecuteReport exec(workloadDesc);
        CSR matA,matB,matC;

        exec.timmer.measure("loadCSR",0,1,[&]{
            matA = safeLoadCSR(descA);
            matB = safeLoadCSR(descB);
        });


        exec.timmer.selfAssistMeasure("SpGemm",1,repeat,[&](TimmerHelper& timmer){
            matC = fun(timmer,matA,matB);
        });

        MatrixDesc updatedDescC;
        exec.timmer.measure("writeCSR",0,1,[&](){
            updatedDescC = safeWriteCSR(descC,matC);
        });
        exec.outputs["matrixC"] = updatedDescC;
        return exec;
    }

    template <class Callable, class CSR2X, class X2CSR>
    ExecuteReport bench(int repeat, CSR2X csr2x,Callable fun, X2CSR x2csr){
        ExecuteReport exec(workloadDesc);
        CSR matA,matB,matC;

        using XFormat = decltype(csr2x(matA));
        XFormat xMa,xMb,xMc;

        exec.timmer.measure("loadCSR",0,1,[&]{
            matA = safeLoadCSR(descA);
            matB = safeLoadCSR(descB);
        });

        exec.timmer.measure("convertCSR2X",0,1,[&](){
            xMa = csr2x(matA);
            xMb = csr2x(matB);
        });

        exec.timmer.measure("SpGemm",1,repeat,[&](){
            xMc = fun(xMa,xMb);
        });

        exec.timmer.measure("convertX2CSR",0,1,[&](){
            matC = x2csr(xMc);
        });

        MatrixDesc updatedDescC;
        exec.timmer.measure("writeCSR",0,1,[&](){
            updatedDescC = safeWriteCSR(descC,matC);
        });
        exec.outputs["matrixC"] = updatedDescC;
        return exec;
    }

};