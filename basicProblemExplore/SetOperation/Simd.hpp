#pragma once
#include <stdint.h>
#include "SpSpInst/SpSpInterface.hpp"

using namespace SpSpInst;
const auto limit = pack(Limit(cpu.logV,{OpSrc::B,Delta::NotEqual}));

template<SpSpEnum::PolicyStruct policy>
struct SetOp{
  static constexpr SpSpEnum::PolicyStruct _policy = policy;
  static constexpr auto getLimitOp2 = pack(GetLimitOp2(ForceEq::Yes,policy.eagerMask,{{Next::Epsilon,Next::Inf},{Next::Epsilon,Next::Inf}}));

  static int op(uint32_t *a,int lenA, uint32_t *b, int lenB, uint32_t *c){
    int pa,pb,pc;
    pa = pb = pc = 0;
    while(pa < lenA || pb < lenB){
      auto predA = whilelt(pa,lenA);
      auto predB = whilelt(pb,lenB);
      auto bigCmp = InitBigCmp(limit,predA,predB);
      
      auto idxA = load(predA,a+pa);
      auto idxB = load(predB,b+pb);
      
      bigCmp = KeyCombine(bigCmp,idxA,idxB);
      auto matRes = Match(bigCmp,policy.policyMask.A,policy.policyMask.B);
      auto outLimit = GetLimit(bigCmp,policy.simPolicyMask,getLimitOp2);

      auto unpackedLimit = unpack<Limit>(outLimit);

      int genC = unpackedLimit.generate.A; // When ForceEq::Yes, unpackedLimit.generate.A == unpackedLimit.generate.B

      auto predLow = whilelt(0,genC);
      auto idxLeftA = SEPermute<SEPart{OpSrc::A,LRPart::Left}>(matRes,idxA,SEPair{0U,0U});
      auto idxLeftB = SEPermute<SEPart{OpSrc::B,LRPart::Left}>(matRes,idxB,SEPair{0U,0U});
      auto idxLeft = simd_or(predLow,idxLeftA,idxLeftB);
      store(predLow,c + pc, idxLeft);

      if(genC > cpu.v){
        auto predHigh = whilelt(cpu.v,genC);
        auto idxRightA = SEPermute<SEPart{OpSrc::A,LRPart::Right}>(matRes,idxA,SEPair{0u,0u});
        auto idxRightB = SEPermute<SEPart{OpSrc::B,LRPart::Right}>(matRes,idxB,SEPair{0u,0u});
        auto idxRight = simd_or(predHigh,idxRightA,idxRightB);
        store(predHigh,c + pc + cpu.v, idxRight);
      }

      pa += unpackedLimit.consume.A;
      pb += unpackedLimit.consume.B;
      pc += genC;
    }
    return pc;
  }
};