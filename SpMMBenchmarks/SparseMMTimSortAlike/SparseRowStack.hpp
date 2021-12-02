#pragma once
#include <vector>
#include "SparseMatTool/format.hpp"
#include "VectorUtils.hpp"
#include "timmingDebug.hpp"
#include "fixGem5Bug.hpp"
struct EntryInfo
{
    Size_t start;
    Size_t length;
    Size_t pos;
};

// Size_t merger(Idx *__restrict a_pidx, Val *__restrict a_pval, Size_t a_len,
//               Idx *__restrict b_pidx, Val *__restrict b_pval, Size_t b_len,
//               Idx *__restrict c_pidx, Val *__restrict c_pval);
// // Size_t merger(Idx * a_pidx, Val * a_pval, Size_t a_len,
// //               Idx * b_pidx, Val * b_pval, Size_t b_len,
// //               Idx * c_pidx, Val * c_pval);

template<class Callable>
struct SparseRowStack
{
    Callable merge;
    SparseRowStack(Callable merge):merge(merge){}
    Buffer<EntryInfo> infoStack;
    Buffer<Val> valueStack[2];
    Buffer<Idx> idxStack[2];

    // Temp working space
    // Buffer<Val> tempValue;
    // Buffer<Idx> tempIdx;

    Idx stackDepth = 0;

    void reserve(NNZIdx a_rowNNZMax, Size_t c_rowNNZMax)
    {
        // Dirty hack: Increment c_rowNNZMax by 1 
        // Just to avoid the potential out of range bug of this merger impl
        //  For better performance
        c_rowNNZMax += 1;  
        infoStack.resize(128);
        valueStack[0].resize(avoidGhostZone(4*c_rowNNZMax));
        idxStack[0].resize(avoidGhostZone(4*c_rowNNZMax));
        valueStack[1].resize(avoidGhostZone(4*c_rowNNZMax));
        idxStack[1].resize(avoidGhostZone(4*c_rowNNZMax));

    }

    void clear(){
        stackDepth = 0; // This is the only thing to do.
    }

    // Warning: According to the convention
    // The top entry r=1, second entry r=2.
    // r=0 is always invalid!
    EntryInfo &getTop(NNZIdx r)
    {
        return infoStack[stackDepth - r];
    }

    Size_t getFreeSpaceAfter(NNZIdx r)
    {
        if (r <= stackDepth)
        {
            auto info = getTop(r);
            return info.start + info.length;
        }
        else
        {
            // Reach the bottom of stack
            return 0;
        }
    }

    void mergeRandRPlus1(NNZIdx r)
    {
        StartTimmer(reduceMain);

        StartTimmer(reducePrepare);
        auto infoA = getTop(r + 1);
        auto whoA = infoA.pos;
        auto infoB = getTop(r);
        auto whoB = infoB.pos;
        Idx * __restrict__ a_pidx = idxStack[whoA].data() + infoA.start;
        Val * __restrict__ a_pval = valueStack[whoA].data() + infoA.start;
        Size_t a_len = infoA.length;
        Idx * __restrict__ b_pidx = idxStack[whoB].data() + infoB.start;
        Val * __restrict__ b_pval = valueStack[whoB].data() + infoB.start;
        Size_t b_len = infoB.length;

        Size_t newStart = getFreeSpaceAfter(r + 2);
        auto whoC = 1 - whoA;
        Idx * __restrict__ c_pidx = idxStack[whoC].data() + newStart;
        Val * __restrict__ c_pval = valueStack[whoC].data() + newStart;

        Size_t c_len;
        RecordTimmer(reducePrepare);
        StartTimmer(merge);
        c_len = merge(a_pidx, a_pval, a_len,
                    b_pidx, b_pval, b_len,
                    c_pidx, c_pval); // 1/3
        RecordTimmer(merge);


        StartTimmer(reduceMisc);
        // Update the info stack
        // Step 1) [r+1] <- new entry C
        getTop(r + 1) = {newStart, c_len, whoC};
        // Step 2) [r] <- [r-1], [r-1] <- [r-2], ... [2] <- [1]
        while (r >= 2)
        {
            getTop(r) = getTop(r - 1);
            r--;
        }
        // Step 3)  [1] <- removed
        stackDepth--;
        RecordTimmer(reduceMisc);
        
        RecordTimmer(reduceMain);
    }

    void reduce(bool forceMerge = false, int stopAt = 1)
    {
        while (stackDepth > stopAt)
        {
            StartTimmer(analysisTimR1R2R3);
            Size_t r1 = getTop(1).length;
            Size_t r2 = getTop(2).length;

            bool needMerge = forceMerge || r1 >= r2;
            bool mergeR2R3 = false;
            if (stackDepth >= 3)
            {
                Size_t r3 = getTop(3).length;
                needMerge |= r1 + r2 >= r3;
                mergeR2R3 = r1 > r3;

                if (stackDepth >= 4)
                {
                    Size_t r4 = getTop(4).length;
                    needMerge |= r2 + r3 >= r4;
                }
            }
            RecordTimmer(analysisTimR1R2R3);
            // Do the merge when 
            StartTimmer(execMergeR1R2R3);
            if (needMerge)
            {
                if (mergeR2R3)
                {
                    // merge r2,r3
                    mergeRandRPlus1(2);
                }
                else
                {
                    // merge r1,r2
                    mergeRandRPlus1(1);
                }
            } else {
                break;
            }
            RecordTimmer(execMergeR1R2R3);
        }
    }

    Size_t exportResultTo(Idx* __restrict__ dest_pidx, Val* __restrict__ dest_pval){
        if (stackDepth == 2){
            StartTimmer(reducePrepare);
            auto infoA = getTop(1 + 1);
            auto whoA = infoA.pos;
            auto infoB = getTop(1);
            auto whoB = infoB.pos;
            Idx * __restrict__ a_pidx = idxStack[whoA].data() + infoA.start;
            Val * __restrict__ a_pval = valueStack[whoA].data() + infoA.start;
            Size_t a_len = infoA.length;
            Idx * __restrict__ b_pidx = idxStack[whoB].data() + infoB.start;
            Val * __restrict__ b_pval = valueStack[whoB].data() + infoB.start;
            Size_t b_len = infoB.length;

            Idx *c_pidx = dest_pidx;
            Val *c_pval = dest_pval;
            Size_t c_len;
            RecordTimmer(reducePrepare);
            StartTimmer(finalReduce);            
            StartTimmer(merge);
            c_len = merge(a_pidx, a_pval, a_len,
                        b_pidx, b_pval, b_len,
                        c_pidx, c_pval);
            RecordTimmer(merge);   
            RecordTimmer(finalReduce);
            return c_len;         

        } else {
            auto infoA = getTop(1);
            auto whoA = infoA.pos;
            Idx * __restrict__ a_pidx = idxStack[whoA].data() + infoA.start;
            Val * __restrict__ a_pval = valueStack[whoA].data() + infoA.start;

            StartTimmer(copyBack);
            vectorCopy(a_pidx,a_pval,infoA.length,
                dest_pidx,dest_pval);
            RecordTimmer(copyBack);
            return infoA.length;
        }
    }

    void allocateOnTop(Size_t length)
    {
        Size_t newStart = getFreeSpaceAfter(1);
        infoStack[stackDepth++] = {newStart, length,0};
    }
};
