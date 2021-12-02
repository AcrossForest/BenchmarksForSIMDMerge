#pragma once
#include <algorithm>
#include <stdint.h>
#include "TagType.hpp"
namespace OptimizedTim{

struct ScalarToolkit{

template<bool doMul>
inline static void move(OptionalFloat<doMul> mul,int len, 
    const uint32_t * __restrict idxFrom, uint32_t * __restrict idxTo,
    const float * __restrict valFrom, float * __restrict valTo 
    ){
    std::copy(idxFrom,idxFrom+len,idxTo);
    if constexpr(doMul){
      std::transform(valFrom,valFrom+len,valTo,[mul](float b){return mul.v*b;});
    } else {
      std::copy(valFrom,valFrom+len,valTo);
    }
}

template<bool doMulA, bool doMulB>
inline static int sparseAdd(
    OptionalFloat<doMulA> mulA, OptionalFloat<doMulB> mulB,
    uint32_t *a,float *va, int lenA, uint32_t *b, float *vb, int lenB, uint32_t *c, float *vc){
  int pa,pb,pc;
  pa = pb = pc = 0;

  while(pa < lenA && pb < lenB){
    auto ia = a[pa];
    auto ib = b[pb];
    if(ia == ib){
        c[pc] = ia;
        vc[pc] = mulA * va[pa] + mulB * vb[pb];        
        pa ++; pb ++; pc ++;
      } else if (ia < ib){
        vc[pc] = mulA * va[pa];
        c[pc++] = a[pa++];
      } else {
        vc[pc] = mulB * vb[pb];
        c[pc++] = b[pb++];
      }
  }
  if(pa < lenA){
    move(mulA,lenA-pa,
        a+pa, c+pc,
        va+pa, vc + pc);
    // std::copy(a+pa,a+lenA,c+pc);
    // std::transform(va+pa,va+lenA,vc+pc,opA);
    pc += lenA - pa;
  }
  if(pb < lenB){
    move(mulB,lenB-pb,
        b+pb, c+pc,
        vb+pb, vc + pc);
    // std::copy(b+pb,b+lenB,c+pc);
    // std::transform(vb+pb,vb+lenB,vc+pc,opB);
    pc += lenB - pb;
  }
  return pc;
}



// inline static float idOp(float a){
//     return a;
// }

// inline static auto mkMulOp(float a){
//     return [a](float b){
//         return a*b;
//     };
// }

};

}