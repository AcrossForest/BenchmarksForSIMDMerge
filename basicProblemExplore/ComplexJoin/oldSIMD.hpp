#pragma once
#include <stdint.h>
#include "SpSpInst/SpSpInterface.hpp"




using namespace SpSpInst;

inline const uint64_t longLimitB = pack(Limit{cpu.logV,{OpSrc::B,Delta::NotEqual}});

int longAdd(int * __restrict fromA, float * __restrict fromValA, int lenA, 
    int * __restrict fromB, float * __restrict fromValB, int lenB, 
    int * __restrict toC, float * __restrict toValC){
    int pa,pb,pc;
    pa = pb = pc = 0;

    static constexpr uint64_t getLimitOp2 = pack(GetLimitOp2{ForceEq::Yes,PolicyOR.eagerMask,{{Next::Epsilon,Next::Inf},{Next::Epsilon,Next::Inf}}}); 

    while(pa < lenA || pb < lenB){
        auto predA = whilelt(pa,lenA);
        auto predB = whilelt(pb,lenB);
        auto idxA = load(predA,fromA+pa);
        auto idxB = load(predB,fromB+pb);
        auto valA = load(predA,fromValA+pa);
        auto valB = load(predB,fromValB+pb);

        VBigCmp bigCmp = InitBigCmp(longLimitB,predA,predB);
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
        auto valLeft = simd_add(predLeft,valLeftA,valLeftB);

        store(predLeft,toC + pc,idxLeft);
        store(predLeft,toValC + pc,valLeft);

        if(genC > cpu.v){
            auto predRight = whilelt(cpu.v,genC);
            auto idxRightA = SEPermute<SEPart{OpSrc::A,LRPart::Right}>(matRes,idxA,SEPair{0u,0u});
            auto idxRightB = SEPermute<SEPart{OpSrc::B,LRPart::Right}>(matRes,idxB,SEPair{0u,0u});
            auto idxRight = simd_or(predRight,idxRightA,idxRightB);
            auto valRightA = SEPermute<SEPart{OpSrc::A,LRPart::Right}>(matRes,valA,SEPair{0,0});
            auto valRightB = SEPermute<SEPart{OpSrc::B,LRPart::Right}>(matRes,valB,SEPair{0,0});
            auto valRight = simd_add(predRight,valRightA,valRightB);
            store(predRight,toC+ + pc +cpu.v,idxRight);
            store(predRight,toValC+ pc + cpu.v,valRight);
        }

        pa += unpackLimit.consume.A;
        pb += unpackLimit.consume.B;
        pc += unpackLimit.generate.A;
        // std::cout << genC << std::endl;
    }
    return pc;
}


int longMul(int *fromA, float *fromValA, int lenA, int *fromB, float *fromValB, int lenB, int *toC, float *toValC){
    int pa,pb,pc;
    pa = pb = pc = 0;

    static constexpr uint64_t getLimitOp2 = pack(GetLimitOp2{ForceEq::Yes,PolicyAND.eagerMask,{{Next::Epsilon,Next::Inf},{Next::Epsilon,Next::Inf}}}); 

    while(pa < lenA || pb < lenB){
        auto predA = whilelt(pa,lenA);
        auto predB = whilelt(pb,lenB);
        auto idxA = load(predA,fromA+pa);
        auto idxB = load(predB,fromB+pb);
        auto valA = load(predA,fromValA+pa);
        auto valB = load(predB,fromValB+pb);

        VBigCmp bigCmp = InitBigCmp(longLimitB,predA,predB);
        bigCmp = KeyCombine(bigCmp,idxA,idxB);
        VMatRes matRes = Match(bigCmp,PolicyAND.policyMask.A,PolicyAND.policyMask.B);
        uint64_t newLimit = GetLimit(bigCmp,PolicyAND.simPolicyMask,getLimitOp2);
        Limit unpackLimit = unpack<Limit>(newLimit);
        int genC = unpackLimit.generate.A;

        if(genC > 0){
            auto predLeft = whilelt(0,genC);
            auto idxLeftA = SEPermute<SEPart{OpSrc::A,LRPart::Left}>(matRes,idxA,SEPair{0u,0u});
            auto idxLeftB = SEPermute<SEPart{OpSrc::B,LRPart::Left}>(matRes,idxB,SEPair{0u,0u});
            auto idxLeft = simd_or(predLeft,idxLeftA,idxLeftB);
            auto valLeftA = SEPermute<SEPart{OpSrc::A,LRPart::Left}>(matRes,valA,SEPair{0.0f,0.0f});
            auto valLeftB = SEPermute<SEPart{OpSrc::B,LRPart::Left}>(matRes,valB,SEPair{0.0f,0.0f});
            auto valLeft = simd_mul(predLeft,valLeftA,valLeftB);

            store(predLeft,toC + pc,idxLeft);
            store(predLeft,toValC + pc,valLeft);
        }

        pa += unpackLimit.consume.A;
        pb += unpackLimit.consume.B;
        pc += unpackLimit.generate.A;
    }
    return pc;
}


