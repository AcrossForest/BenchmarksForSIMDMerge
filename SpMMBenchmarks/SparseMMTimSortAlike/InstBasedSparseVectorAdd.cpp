#include "SparseMatTool/format.hpp"
#include "SpSpInterface/SpSpInterface.hpp"
#include "VectorUtils.hpp"
#include "timmingDebug.hpp"
#include <numeric>
#include <iostream>


Size_t instBasedVectorAdd(
    const Idx *__restrict a_pidx, const Val *__restrict a_pval, Size_t a_len,
    const Idx *__restrict b_pidx, const Val *__restrict b_pval, Size_t b_len,
    Idx *__restrict c_pidx, Val *__restrict c_pval)
{
    Size_t a_used = 0;
    Size_t b_used = 0;
    Size_t c_filled = 0;

    // int vecLen = vectorLen();
    
    InitLocalCounter(loopCount);

    if (a_len > 0 && b_len > 0)
    {
        while (a_used < a_len && b_used < b_len){
            IncCounter(loopCount);
            VecBool pred_a,pred_b;

            pred_a = whilelt(a_used,a_len);
            pred_b = whilelt(b_used,b_len);

            VecIdx idx_a,idx_b;
            VecFlt val_a,val_b;
            idx_a = loadVec(pred_a,a_pidx + a_used);
            idx_b = loadVec(pred_b,b_pidx + b_used);
            val_a = loadVec(pred_a,a_pval + a_used);
            val_b = loadVec(pred_b,b_pval + b_used);
            // It is not implemented in Gem5, and act like an nop and reducing performance...
            // Uncomment it the day you have this instruction on real machine.
            // prefVec<EnumPrefetch::SV_PLDL1STRM>(pred_a,a_pidx + a_used + vecLen);
            // prefVec<EnumPrefetch::SV_PLDL1STRM>(pred_a,a_pval + a_used + vecLen);
            // prefVec<EnumPrefetch::SV_PLDL1STRM>(pred_b,b_pidx + b_used + vecLen);
            // prefVec<EnumPrefetch::SV_PLDL1STRM>(pred_b,b_pval + b_used + vecLen);

            Comp comp = indexCompress(
                idx_a,a_len - a_used,
                idx_b,b_len - b_used);
            
            MatRes matRes = indexMatch<EnumInxMatMethod::OR>(
                comp,EnumEndType::FINISHED,EnumEndType::FINISHED);

            Size_t a_consumed = getLength<EnumGetLen::ConsumedA>(matRes);
            Size_t b_consumed = getLength<EnumGetLen::ConsumedB>(matRes);


            ///////////////////////////////////////////////////
            //              Output to array c
            ////////////////////////////////////////////////

            // Shared declares
            VecIdx idx_c_part_a,idx_c_part_b,idx_c;
            VecFlt val_c_part_a,val_c_part_b,val_c;
            VecBool pred_c;
            Size_t c_consumed;

            // First part
            pred_c = getPred<EnumGetPred::Out0>(matRes);
            c_consumed = getLength<EnumGetLen::OutputLen0>(matRes);
            // First part : index part
            idx_c_part_a = permute<EnumGetPermPart::A0>(matRes,idx_a);
            idx_c_part_b = permute<EnumGetPermPart::B0>(matRes,idx_b);
            idx_c = vector_Or(pred_c,idx_c_part_a,idx_c_part_b);
            storeVec(idx_c,pred_c,c_pidx + c_filled);
            // First part : value part
            val_c_part_a = permute<EnumGetPermPart::A0>(matRes,val_a);
            val_c_part_b = permute<EnumGetPermPart::B0>(matRes,val_b);
            val_c = vector_Add(pred_c,val_c_part_a,val_c_part_b);
            storeVec(val_c,pred_c,c_pval + c_filled);
            // First part: update c_filled
            c_filled += c_consumed;

            // second part
            // Jump of no second part
            c_consumed = getLength<EnumGetLen::OutputLen1>(matRes);
            if(c_consumed != 0){
                pred_c = getPred<EnumGetPred::Out1>(matRes);
                // Second part : index part
                idx_c_part_a = permute<EnumGetPermPart::A1>(matRes,idx_a);
                idx_c_part_b = permute<EnumGetPermPart::B1>(matRes,idx_b);
                idx_c = vector_Or(pred_c,idx_c_part_a,idx_c_part_b);
                storeVec(idx_c,pred_c,c_pidx + c_filled);
                // Second part: value part
                val_c_part_a = permute<EnumGetPermPart::A1>(matRes,val_a);
                val_c_part_b = permute<EnumGetPermPart::B1>(matRes,val_b);
                val_c = vector_Add(pred_c,val_c_part_a,val_c_part_b);
                storeVec(val_c,pred_c,c_pval + c_filled);
                // Second part : update c_filled
                c_filled += c_consumed;
            }


            // printf("%lu, %lu\n",a_used,b_used);
            a_used += a_consumed;
            b_used += b_consumed;

            // std::cout << "a_consumed: " << a_consumed << "\t b_consumed:" << b_consumed << std::endl;


        }
        AccumulateCounter(totalLoopCount,loopCount);
        // totalLoopcount += loopCount;
        // totalProccessed += proccessed;
        AccumulateCounter(totalProccessed, std::min(a_len,b_len));
#ifdef __MeasureSpSpVecEfficiency__
        std::cout << "Loop Count: " << loopCount
            << "\tA_len: " << a_len << "\tB:len: " << b_len 
            << "\t Efficiency: " << double()/loopCount 
            << std::endl;
#endif
    }
    if (a_used < a_len)
    {
        vectorCopy(
            a_pidx+a_used, a_pval+ a_used, a_len - a_used,
            c_pidx + c_filled, c_pval + c_filled
        );
        // std::copy(a_pidx + a_used, a_pidx + a_len, c_pidx + c_filled);
        // std::copy(a_pval + a_used, a_pval + a_len, c_pval + c_filled);
        c_filled += a_len - a_used;
        a_used = a_len;
    }
    if (b_used < b_len)
    {
        vectorCopy(
            b_pidx + b_used, b_pval + b_used, b_len - b_used,
            c_pidx + c_filled, c_pval + c_filled
        );
        // std::copy(b_pidx + b_used, b_pidx + b_len, c_pidx + c_filled);
        // std::copy(b_pval + b_used, b_pval + b_len, c_pval + c_filled);
        c_filled += b_len - b_used;
        b_used = b_len;
    }
    return c_filled;
}
