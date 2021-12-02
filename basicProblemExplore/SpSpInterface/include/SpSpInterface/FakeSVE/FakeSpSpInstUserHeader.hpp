#pragma once
#include <stdint.h>
#include <vector>
#include "SpSpInterface/SpSpEnums.hpp"
#include "SpSpInterface/FakeSVE/FakeImplInternal.hpp"

#ifndef SPSP_FAKE_MACHINE_VECLEN
#define SPSP_FAKE_MACHINE_VECLEN 64
#endif

namespace FakeSPSP
{
    using namespace SPSPEnum;

    using ScaIdx = uint32_t;
    using ScaInt = int32_t;
    using ScaFlt = float;

    template <class T>
    using RegVecType = std::vector<T>;

    using VecIdx = RegVecType<ScaIdx>;
    using VecInt = RegVecType<ScaInt>;
    using VecFlt = RegVecType<ScaFlt>;
    using VecBool = RegVecType<uint8_t>;

    using Comp = RegVecType<uint32_t>;
    using MatRes = RegVecType<uint32_t>;
    using SSPERM = RegVecType<uint32_t>;

    const int VecLen = SPSP_FAKE_MACHINE_VECLEN;

///////////////////////////////////////////////////////////
//                  SVE standard instructions
/////////////////////////////////////////////////////////


    inline VecBool whilelt(const ScaIdx& start, const ScaIdx& end){
        VecBool out(VecLen,0);
        for(int i=0; i<VecLen; ++i){
            out[i] = start + i < end;
        }
        return out;
        //return svwhilelt_b32(start,end);
    }

    template <class ScaType>
    inline RegVecType<ScaType> loadVec(const VecBool& pred,const ScaType* base){
        RegVecType<ScaType> out(VecLen,0);
        for(int i=0; i<VecLen; ++i){
            if(pred[i]){
                out[i] = base[i];
            }
        }
        return out;
        // return svld1(pred,base);
    }

    template <EnumPrefetch perfType,class ScaType>
    inline void prefVec(VecBool pred, const ScaType* base){
    }

    inline int vectorLen(){
        return VecLen;
    }

    template <class ScaType>
    inline void storeVec(const RegVecType<ScaType>& data, const VecBool& pred, ScaType* base){
        for(int i=0; i<VecLen; ++i){
            if(pred[i]){
                base[i] = data[i];
            }
        }
        // svst1(pred,base,data);
    }

    template <class ScaType>
    inline RegVecType<ScaType> vector_Or(const VecBool& pred,
        const RegVecType<ScaType>& op1,
        const RegVecType<ScaType>& op2){
        RegVecType<ScaType> out(VecLen,0);
        for(int i=0; i<VecLen; ++i){
            if(pred[i]){
                out[i] = op1[i] | op2[i];
            }
        }
        return out;
        // return svorr_z(pred,op1,op2);
    }

    template <class ScaType>
    inline RegVecType<ScaType> vector_Add(const VecBool& pred,
        const RegVecType<ScaType>& op1,
        const RegVecType<ScaType>& op2){
        RegVecType<ScaType> out(VecLen,0);
        for(int i=0; i<VecLen; ++i){
            if(pred[i]){
                out[i] = op1[i] + op2[i];
            }
        }
        return out;
        // return svadd_z(pred,op1,op2);
    }

    template <class ScaType>
    inline RegVecType<ScaType> vector_Mul_scalar(VecBool pred,
        const RegVecType<ScaType>& op1,
        ScaType op2){
        RegVecType<ScaType> out(VecLen,0);
        for(int i=0; i<VecLen; ++i){
            if(pred[i]){
                out[i] = op1[i] + op2;
            }
        }
        return out;
    }

///////////////////////////////////////////////////////////
//                  SPSP instructions
/////////////////////////////////////////////////////////
    inline uint64_t pack(ScaIdx a, ScaInt b)
    {
        return (uint64_t(a) << 32) | b;
    }

    inline Comp indexCompress(const VecIdx& a, uint32_t lenA,
                              const VecIdx& b, uint32_t lenB)
    {
        uint64_t lenab = pack(lenA, lenB);
        return SPSPINST::Machine(VecLen).indexCompressionInitM1(
            a, b, lenab);
    }
    inline Comp indexCompress(const Comp& old, const VecIdx& a,
                              const VecIdx& b)
    {
        return SPSPINST::Machine(VecLen).indexCompression(
            old, a, b);
    }

    template <EnumInxMatMethod method>
    inline MatRes indexMatch(const Comp& comp, EnumEndType endA, EnumEndType endB)
    {
        return SPSPINST::Machine(VecLen).indexMatchM2(
            comp, SPSPINST::EndType(endA), SPSPINST::EndType(endB),
            SPSPINST::IndexMatchMethod(method));
        // return svSpSpIndexMatchM2(
        //     comp, uint32_t(endA), uint32_t(endB), uint32_t(method));
    }

    template <EnumGetPermPart part,class ScaType>
    inline RegVecType<ScaType> permute(const MatRes &matRes, const RegVecType<ScaType>& src)
    {
        return SPSPINST::Machine(VecLen).customPermute(
            matRes, src, SPSPINST::MatchResultPart(part));
        // return svSpSpCustPerm_u32(matRes, src, uint32_t(part));
        // return svSpSpCustPerm_s32(matRes, src, uint32_t(part));
        // return svSpSpCustPerm_f32(matRes, src, uint32_t(part));
    }


    template <EnumGetPred predPart>
    inline VecBool getPred(const MatRes& matRes)
    {
        return SPSPINST::Machine(VecLen).getPredicate(
            matRes, SPSPINST::MatchPredResult(predPart));
        // return svSpSpGetPred(matRes, uint32_t(predPart));
    }

    template <EnumGetLen lenPart>
    inline ScaIdx getLength(const MatRes& matRes){
        return SPSPINST::Machine(VecLen).getLength(
            matRes,SPSPINST::MatchResultLength(lenPart)
        );
        // return svSpSpGetLength(matRes,uint32_t(lenPart));
    }

    inline SSPERM singleSideSort(const VecIdx& idx, const VecBool& pred)
    {
        return SPSPINST::Machine(VecLen).singleSideSortInit(
            idx, pred);
        // return svSpSpSingleSideSortInit(idx, pred);
    }

    inline VecIdx singleSideSort(const SSPERM& prepermute, const VecIdx& idx, const VecBool & pred)
    {
        return SPSPINST::Machine(VecLen).singleSideSort(
            prepermute, idx, pred);
        // return svSpSpSingleSideSort(prepermute, idx, pred);
    }

    inline VecBool diff(const VecIdx & idx, const ScaIdx& idx0)
    {
        return SPSPINST::Machine(VecLen).diff(idx, idx0);
        // return svSpSpDiff(idx, idx0);
    }

    inline VecIdx encoding(const VecBool& pred)
    {
        return SPSPINST::Machine(VecLen).encoding(pred);
        // return svSpSpEncoding(pred);
    }

    template <EnumGetPermPart part>
    inline ScaIdx getLastActiveElem(const MatRes& matRes, const VecIdx& src)
    {
        return SPSPINST::Machine(VecLen).getLastActiveElem(
            matRes, src, SPSPINST::MatchResultPart(part));
        // return svSpSpCustLastActElem(matRes, src, uint32_t(part));
    }

    template <class ScaType>
    inline RegVecType<ScaType> ringLoad(const VecBool& pred, ScaType *base,
                                        ScaIdx offset, ScaIdx boundary)
    {
        RegVecType<ScaType> readout(VecLen);
        for (int i = 0; i < VecLen; ++i)
        {
            if (pred[i] != 0)
            {
                readout[i] = base[(i + offset) % boundary];
            }
        }
        return readout;
        // return svSpSpRingLoad_u32(pred, base, combined);
        // return svSpSpRingLoad_s32(pred, base, combined);
        // return svSpSpRingLoad_f32(pred, base, combined);
    }

    template <class ScaType>
    inline void ringStore(const RegVecType<ScaType>& data, const VecBool& pred, ScaType *base,
                          ScaIdx offset, ScaIdx boundary)
    {
        for (int i = 0; i < VecLen; ++i)
        {
            if (pred[i] != 0)
            {
                base[(i + offset) % boundary] = data[i];
            }
        }
        // return svSpSpRingStore_u32(data, pred, base, combined);
        // return svSpSpRingStore_s32(data, pred, base, combined);
        // return svSpSpRingStore_f32(data, pred, base, combined);
    }

} // namespace FakeSPSP