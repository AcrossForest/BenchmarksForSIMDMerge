#pragma once
#include <stdint.h>
#include "SpSpInst/SpSpInterface.hpp"
namespace OptimizedTim{

inline const uint64_t longLimit = SpSpInst::pack(SpSpInst::Limit{SpSpInst::cpu.logV,
    {SpSpInst::OpSrc::B,SpSpInst::Delta::NotEqual}});

struct SIMDToolkit{


template<bool doMul>
inline static SpSpInst::VReg<float> conditionalMul(SpSpInst::VecBool pred, SpSpInst::VReg<float> vec, OptionalFloat<doMul> mul){
    if constexpr(doMul){
        return SpSpInst::simd_mul_vs(pred,vec,mul.v);
    } else {
        return vec;
    }
}

template<bool doMul>
inline static void move(OptionalFloat<doMul> mul,int len, 
    const uint32_t * __restrict idxFrom, uint32_t * __restrict idxTo,
    const float * __restrict valFrom, float * __restrict valTo 
    ){
    using namespace SpSpInst;
    int p = 0;
    while(p < len){
        auto pred = whilelt(p,len);
        auto idx = load(pred,idxFrom + p);
        auto val = load(pred,valFrom + p);
        store(pred, idxTo + p, idx);
        store(pred, valTo + p, conditionalMul(pred,val,mul));
        p += cpu.v;
    }
}


// inline static auto idOp(SpSpInst::VecBool pred, SpSpInst::VReg<float> vec){
//     return vec;
// }

// inline static auto mkMulOp(float a){
//     using namespace SpSpInst;
//     return [a](VecBool pred,VReg<float> b){
//         return simd_scalar_mul(pred,b,a);
//     };
// }




template<bool doMulA, bool doMulB>
inline static int sparseAdd(
        OptionalFloat<doMulA> mulA, OptionalFloat<doMulB> mulB,
        uint32_t * __restrict fromA, float * __restrict fromValA, int lenA, 
        uint32_t * __restrict fromB, float * __restrict fromValB, int lenB, 
        uint32_t * __restrict toC, float * __restrict toValC){
        using namespace SpSpInst;
        int pa,pb,pc;
        pa = pb = pc = 0;

        static constexpr uint64_t getLimitOp2 = pack(GetLimitOp2{ForceEq::Yes,PolicyOR.eagerMask,{{Next::Epsilon,Next::Inf},{Next::Epsilon,Next::Inf}}}); 

 


        while(pa < lenA && pb < lenB){
            auto predA = whilelt(pa,lenA);
            auto predB = whilelt(pb,lenB);
            auto idxA = load(predA,fromA+pa);
            auto idxB = load(predB,fromB+pb);
            auto valA = load(predA,fromValA+pa);
            auto valB = load(predB,fromValB+pb);

            VBigCmp bigCmp = InitBigCmp(longLimit,predA,predB);
            bigCmp = KeyCombine(bigCmp,idxA,idxB);
            VMatRes matRes = Match(bigCmp,PolicyOR.policyMask.A,PolicyOR.policyMask.B);
            uint64_t newLimit = GetLimit(bigCmp,PolicyOR.simPolicyMask,getLimitOp2);
            Limit unpackLimit = unpack<Limit>(newLimit);
            int genC = unpackLimit.generate.A;

            auto predLeft = whilelt(0,genC);
            auto idxLeftA = SEPermute<SEPart{OpSrc::A,LRPart::Left}>(matRes,idxA,SEPair{0u,0u});
            auto idxLeftB = SEPermute<SEPart{OpSrc::B,LRPart::Left}>(matRes,idxB,SEPair{0u,0u});
            auto idxLeft = simd_or(predLeft,idxLeftA,idxLeftB);
            auto valLeftA = SEPermute<SEPart{OpSrc::A,LRPart::Left}>(matRes,valA,SEPair{0,0});
            auto valLeftB = SEPermute<SEPart{OpSrc::B,LRPart::Left}>(matRes,valB,SEPair{0,0});

            auto valLeft = simd_add(predLeft,
                conditionalMul(predLeft,valLeftA,mulA),
                conditionalMul(predLeft,valLeftB,mulB));

            store(predLeft,toC + pc,idxLeft);
            store(predLeft,toValC + pc,valLeft);

            if(genC > cpu.v){
                auto predRight = whilelt(cpu.v,genC);
                auto idxRightA = SEPermute<SEPart{OpSrc::A,LRPart::Right}>(matRes,idxA,SEPair{0u,0u});
                auto idxRightB = SEPermute<SEPart{OpSrc::B,LRPart::Right}>(matRes,idxB,SEPair{0u,0u});
                auto idxRight = simd_or(predRight,idxRightA,idxRightB);
                auto valRightA = SEPermute<SEPart{OpSrc::A,LRPart::Right}>(matRes,valA,SEPair{0,0});
                auto valRightB = SEPermute<SEPart{OpSrc::B,LRPart::Right}>(matRes,valB,SEPair{0,0});
                auto valRight = simd_add(predRight,
                    conditionalMul(predRight,valRightA,mulA),
                    conditionalMul(predRight,valRightB,mulB));
                store(predRight,toC+ + pc +cpu.v,idxRight);
                store(predRight,toValC+ pc + cpu.v,valRight);
            }

            pa += unpackLimit.consume.A;
            pb += unpackLimit.consume.B;
            pc += unpackLimit.generate.A;
            // std::cout << genC << std::endl;
        }

        if(pa < lenA){
            move(mulA, lenA - pa,
                fromA + pa, toC + pc,
                fromValA + pa, toValC + pc
            );
            pc += lenA - pa;
        }
        if(pb < lenB){
            move(mulB, lenB - pb,
                fromB + pb, toC + pc,
                fromValB + pb, toValC + pc
            );
            pc += lenB - pb;
        }
        return pc;
    }


// inline static auto idOp(SpSpInst::VecBool pred, SpSpInst::VReg<float> vec){
//     return vec;
// }

// inline static auto mkMulOp(float a){
//     using namespace SpSpInst;
//     return [a](VecBool pred,VReg<float> b){
//         return simd_scalar_mul(pred,b,a);
//     };
// }



};

}