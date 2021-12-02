#pragma once
#include <arm_sve.h>
#include "SpSpInterface/SpSpEnums.hpp"

namespace SVEWrapper{
    using VecIdx = svuint32_t;
    using VecInt = svint32_t;
    using VecFlt = svfloat32_t;
    using VecBool = svbool_t;

    using ScaIdx = uint32_t;
    using ScaInt = int32_t;
    using ScaFlt = float32_t;

    template <class T>
    struct RegVecTypeStruct{
        using VecType = void;
    };


    template <>
    struct RegVecTypeStruct<ScaIdx>{
        using VecType = VecIdx;
    };
    template <>
    struct RegVecTypeStruct<ScaInt>{
        using VecType = VecInt;
    };
    template <>
    struct RegVecTypeStruct<VecFlt>{
        using VecType = VecFlt;
    };
    template <>
    struct RegVecTypeStruct<float>{
        using VecType = VecFlt;
    };
    template <>
    struct RegVecTypeStruct<bool>{
        using VecType = VecBool;
    };

    template <class T>
    using RegVecType = RegVecTypeStruct<T>::VecType;




    // Wrap some useful offitial instructions

    inline VecBool whilelt(ScaIdx start, ScaIdx end){
        return svwhilelt_b32(start,end);
    }

    template <class ScaType>
    inline RegVecType<ScaType> loadVec(VecBool pred,const ScaType* base){
        return svld1(pred,base);
    }

    template <class ScaType>
    inline void storeVec(RegVecType<ScaType> data, VecBool pred, ScaType* base){
        svst1(pred,base,data);
    }

    using SPSPEnum::EnumPrefetch;
    template <EnumPrefetch perfType, class ScaType>
    inline void prefVec(VecBool pred, const ScaType* base){
        svprfw(pred,base,svprfop(perfType));
    }

    inline int vectorLen(){
        return svcntw();
    }

    // This way is more nature, unfortunatly it don't work
    // Function resolution with type trait seems not working
    // Don't work: idx_c = vector_Or(pred_c,idx_c_part_a,idx_c_part_b);
    // Pass:  idx_c = vector_Or<ScaIdx>(pred_c,idx_c_part_a,idx_c_part_b);
    // template <class ScaType>
    // RegVecType<ScaType> vector_Or(VecBool pred,
    //     RegVecType<ScaType> op1,
    //     RegVecType<ScaType> op2){
    //     return svorr_z(pred,op1,op2);
    // }

    // This work 
    template <class VecType>
    inline VecType vector_Or(VecBool pred,
        VecType op1,
        VecType op2){
        return svorr_x(pred,op1,op2);
    }


    template <class VecType>
    inline VecType vector_Add(VecBool pred,
        VecType op1,
        VecType op2){
        return svadd_x(pred,op1,op2);
    }

    template <class ScaType>
    inline RegVecType<ScaType> vector_Mul_scalar(VecBool pred,
        RegVecType<ScaType> op1,
        ScaType op2){
        return svmul_x(pred,op1,op2);
    }



}

namespace SPSP
{
    using namespace SVEWrapper;
    using namespace SPSPEnum;



    using Comp = svuint32_t;
    using MatRes = svuint32_t;
    using SSPERM = svuint32_t;

    


    inline Comp indexCompress(VecIdx a, uint32_t lenA,
                              VecIdx b, uint32_t lenB)
    {
        uint64_t lenab = svSpSpPack(lenA, lenB);
        return svSpSpIndexCompressInitM1(a, b, lenab);
    }
    inline Comp indexCompress(Comp old, VecIdx a,
                              VecIdx b)
    {
        return svSpSpIndexCompression(old, a, b);
    }

    template <EnumInxMatMethod method>
    inline MatRes indexMatch(Comp comp, EnumEndType endA, EnumEndType endB)
    {
        return svSpSpIndexMatchM2(
            comp, uint32_t(endA), uint32_t(endB), uint32_t(method));
    }

    template <EnumGetPermPart part>
    inline VecIdx permute(MatRes matRes, VecIdx src)
    {
        return svSpSpCustPerm_u32(matRes, src, uint32_t(part));
    }
    template <EnumGetPermPart part>
    inline VecInt permute(MatRes matRes, VecInt src)
    {
        return svSpSpCustPerm_s32(matRes, src, uint32_t(part));
    }
    template <EnumGetPermPart part>
    inline VecFlt permute(MatRes matRes, VecFlt src)
    {
        return svSpSpCustPerm_f32(matRes, src, uint32_t(part));
    }

    template <EnumGetPred predPart>
    inline VecBool getPred(MatRes matRes)
    {
        return svSpSpGetPred(matRes, uint32_t(predPart));
    }

    template <EnumGetLen lenPart>
    inline ScaIdx getLength(MatRes matRes){
        return svSpSpGetLength(matRes,uint32_t(lenPart));
    }

    inline SSPERM singleSideSort(VecIdx idx, VecBool pred)
    {
        return svSpSpSingleSideSortInit(idx, pred);
    }

    inline VecIdx singleSideSort(SSPERM prepermute, VecIdx idx, VecBool pred)
    {
        return svSpSpSingleSideSort(prepermute, idx, pred);
    }

    inline VecBool diff(VecIdx idx, ScaIdx idx0)
    {
        return svSpSpDiff(idx, idx0);
    }

    inline VecIdx encoding(VecBool pred)
    {
        return svSpSpEncoding(pred);
    }

    template <EnumGetPermPart part>
    inline ScaIdx getLastActiveElem(MatRes matRes, VecIdx src)
    {
        return svSpSpCustLastActElem(matRes, src, uint32_t(part));
    }

    inline VecIdx ringLoad(VecBool pred, ScaIdx *base, ScaIdx offset,
                           ScaIdx boundary)
    {
        uint64_t combined = svSpSpPack(offset, boundary);
        return svSpSpRingLoad_u32(pred, base, combined);
    }
    inline VecInt ringLoad(VecBool pred, ScaInt *base, ScaIdx offset,
                           ScaIdx boundary)
    {
        uint64_t combined = svSpSpPack(offset, boundary);
        return svSpSpRingLoad_s32(pred, base, combined);
    }
    inline VecFlt ringLoad(VecBool pred, ScaFlt *base, ScaIdx offset,
                           ScaIdx boundary)
    {
        uint64_t combined = svSpSpPack(offset, boundary);
        return svSpSpRingLoad_f32(pred, base, combined);
    }

    inline void ringStore(VecIdx data, VecBool pred, ScaIdx *base, ScaIdx offset,
                          ScaIdx boundary)
    {
        uint64_t combined = svSpSpPack(offset, boundary);
        return svSpSpRingStore_u32(data,pred, base, combined);
    }
    inline void ringStore(VecInt data, VecBool pred, ScaInt *base, ScaIdx offset,
                          ScaIdx boundary)
    {
        uint64_t combined = svSpSpPack(offset, boundary);
        return svSpSpRingStore_s32(data,pred, base, combined);
    }
    inline void ringStore(VecFlt data, VecBool pred, ScaFlt *base, ScaIdx offset,
                          ScaIdx boundary)
    {
        uint64_t combined = svSpSpPack(offset, boundary);
        return svSpSpRingStore_f32(data,pred, base, combined);
    }

} // namespace SPSP