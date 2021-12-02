#pragma once
#include <memory>
#include <stdint.h>
#include "SparseMatTool/format.hpp"
#include "fixGem5Bug.hpp"
#include "TagType.hpp"
#include "measureMerge.hpp"

namespace OptimizedTim{


struct EntryInfo{
    uint32_t start,length;
    uint8_t pos; // 0,1 are buffers, "2" is external position
    bool needMulWieght;
    uint32_t remoteStart;
    float weight;

    // uint32_t getRealStart(){
    //     return start;
    // }
    uint32_t getRealStart(){
        return needMulWieght ? remoteStart : start;
    }
};

template<class Toolkit>
struct SparseRowStack
{
    SparseRowStack(Toolkit toolkit){}

    Buffer<EntryInfo> infoStack;
    Buffer<float> _valStackHolder[2];
    Buffer<uint32_t> _idxStackHolder[2];

    uint32_t* idxBuffer[3];
    float* valBuffer[3];

    // Temp working space
    // Buffer<float> tempValue;
    // Buffer<uint32_t> tempIdx;

    static constexpr int stackBottom = 0;
    int stackTop = 0;

    void reserve(size_t c_rowNNZMax, const uint32_t *c_col_idxs, const float *c_val)
    {
        infoStack.resize(128);
        for(int i=0; i<2; ++i){
            _idxStackHolder[i].resize(avoidGhostZone(4 * c_rowNNZMax));
            _valStackHolder[i].resize(avoidGhostZone(4 * c_rowNNZMax));
            idxBuffer[i]=_idxStackHolder[i].data();
            valBuffer[i]=_valStackHolder[i].data();
        }
        idxBuffer[2] = const_cast<uint32_t*>(c_col_idxs);
        valBuffer[2] = const_cast<float*>(c_val);

        infoStack[stackBottom] = {0,0,0,false,0, 0.0f};
        stackTop = stackBottom;
    }

    void clear(){
        stackTop = stackBottom; // This is the only thing to do.
    }

    auto getStackSize(){
        return stackTop - stackBottom;
    }

    bool isValid(NNZIdx r){
        return r < getStackSize();
    }

    EntryInfo &getTop(NNZIdx r)
    {
        return infoStack[stackTop - r];
    }

    auto getFreeSpaceAfter(NNZIdx r)
    {
        auto info = getTop(r);
        return info.start + info.length;
    }


    // merge stack position r and r+1
    uint32_t mergeAndOutputTo(NNZIdx r, uint32_t* c_pidx, float * c_pval)
    {

        auto infoA = getTop(r + 1);
        auto whoA = infoA.pos;
        auto startA = infoA.getRealStart();
        auto infoB = getTop(r);
        auto whoB = infoB.pos;
        auto startB = infoB.getRealStart();
        uint32_t * __restrict__ a_pidx = idxBuffer[whoA] + startA;
        float * __restrict__ a_pval = valBuffer[whoA] + startA;
        auto a_len = infoA.length;
        uint32_t * __restrict__ b_pidx = idxBuffer[whoB] + startB;
        float * __restrict__ b_pval = valBuffer[whoB] + startB;
        auto b_len = infoB.length;

        Size_t c_len;

        MergeStartTimmer(merge);

        if(infoA.needMulWieght){
            if(infoB.needMulWieght){
                // a need weight, b need weight
                c_len = Toolkit::sparseAdd(
                    OptionalFloat<true>{infoA.weight},OptionalFloat<true>{infoB.weight},
                    a_pidx, a_pval, a_len,
                    b_pidx, b_pval, b_len,
                    c_pidx, c_pval
                );
            } else {
                // a need wieght, b don't
                c_len = Toolkit::sparseAdd(
                    OptionalFloat<true>(infoA.weight),OptionalFloat<false>(),
                    a_pidx, a_pval, a_len,
                    b_pidx, b_pval, b_len,
                    c_pidx, c_pval
                ); 
            }
        } else {
            if(infoB.needMulWieght){
                // a need weight, b need weight
                c_len = Toolkit::sparseAdd(
                    OptionalFloat<false>(),OptionalFloat<true>(infoB.weight),
                    a_pidx, a_pval, a_len,
                    b_pidx, b_pval, b_len,
                    c_pidx, c_pval
                );
            } else {
                // a need wieght, b don't
                c_len = Toolkit::sparseAdd(
                    OptionalFloat<true>(),OptionalFloat<false>(),
                    a_pidx, a_pval, a_len,
                    b_pidx, b_pval, b_len,
                    c_pidx, c_pval
                );
            }

        }

        MergeRecordTimmer(merge);


        return c_len;
    }

    // copy content of entry r out to
    uint32_t copyAndOutputTo(NNZIdx r, uint32_t* c_pidx, float * c_pval){
        auto infoA = getTop(r);
        auto whoA = infoA.pos;
        auto startA = infoA.getRealStart();
        uint32_t * __restrict__ a_pidx = idxBuffer[whoA] + startA;
        float * __restrict__ a_pval = valBuffer[whoA] + startA;
        auto a_len = infoA.length;
        // Toolkit::move(OptionalFloat<false>{},a_len, a_pidx, c_pidx, a_pval,c_pval);
        if(infoA.needMulWieght){
            Toolkit::move(OptionalFloat<true>(infoA.weight),a_len, a_pidx, c_pidx, a_pval,c_pval);
        } else {
            Toolkit::move(OptionalFloat<false>(),a_len, a_pidx, c_pidx, a_pval,c_pval);
        }
        return a_len;
    }
    // merge stack position r and r+1
    void mergeRandRPlus1(NNZIdx r)
    {
        auto newStart = getFreeSpaceAfter(r + 2);
        auto whoC = getTop(r + 1).pos==0? 1:0;
        uint32_t * __restrict__ c_pidx = idxBuffer[whoC] + newStart;
        float * __restrict__ c_pval = valBuffer[whoC] + newStart;

        auto c_len = mergeAndOutputTo(r,c_pidx, c_pval);

        // Update the info stack
        // Step 1) [r+1] <- new entry C
        // getTop(r + 1) = {newStart, c_len, uint8_t(whoC)};
        getTop(r + 1) = {newStart, c_len, uint8_t(whoC), false,0, 0.0f};
        // Step 2) [r] <- [r-1], [r-1] <- [r-2], ... [2] <- [1]
        while (r >= 1)
        {
            getTop(r) = getTop(r - 1);
            r--;
        }
        // Step 3)  [1] <- removed
        stackTop--;
    }

    // reduce the stack. Stop when nothng to merge, or remain only 'remain' entries
    void reduce(bool forceMerge = false, int remain = 1)
    {
        while (getStackSize() > remain)
        {
            Size_t r1 = getTop(0).length;
            Size_t r2 = getTop(1).length;
            auto stackSize = getStackSize();

            bool needMerge = forceMerge || r1 >= r2;
            bool mergeR2R3 = false;
            if (stackSize >= 3)
            {
                Size_t r3 = getTop(2).length;
                needMerge |= r1 + r2 >= r3;
                mergeR2R3 = r1 > r3;

                if (stackSize >= 4)
                {
                    Size_t r4 = getTop(3).length;
                    needMerge |= r2 + r3 >= r4;
                }
            }
            // Do the merge when 
            if (needMerge)
            {
                if (mergeR2R3)
                {
                    // merge r2,r3
                    mergeRandRPlus1(1);
                }
                else
                {
                    // merge r1,r2
                    mergeRandRPlus1(0);
                }
            } else {
                break;
            }
        }
    }

    uint32_t exportResultTo(uint32_t* __restrict__ dest_pidx, float* __restrict__ dest_pval){
        if (getStackSize() == 2){
            return mergeAndOutputTo(0,dest_pidx,dest_pval);
        } else {
            return copyAndOutputTo(0,dest_pidx,dest_pval);
        }
        clear();
    }

    void putOnTop(float weight, uint32_t c_start, uint32_t length){
        auto start = getFreeSpaceAfter(0);
        // Toolkit::move(OptionalFloat<true>{weight},length, 
        //     idxBuffer[2]+c_start, idxBuffer[0]+start,
        //     valBuffer[2]+c_start, valBuffer[0]+start);
        
        ++stackTop;
        // infoStack[stackTop] = {start,length,0};
        infoStack[stackTop] = {start, length, 2, true, c_start , weight};
    }

};
}