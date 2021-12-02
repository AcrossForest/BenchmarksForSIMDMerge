#pragma once
#include <stdint.h>
#include <algorithm>
#include "SpSpInst/SpSpInterface.hpp"

inline int sparse_add(int *a,float *va, int lenA, int *b, float *vb, int lenB, int *c, float *vc){
  int pa,pb,pc;
  pa = pb = pc = 0;

  while(pa < lenA && pb < lenB){
    auto ia = a[pa];
    auto ib = b[pb];
    if(ia == ib){
        c[pc] = ia;
        vc[pc] = va[pa] + vb[pb];
        pa ++; pb ++; pc ++;
      } else if (ia < ib){
        vc[pc] = va[pa];
        c[pc++] = a[pa++];
      } else {
        vc[pc] = vb[pb];
        c[pc++] = b[pb++];
      }
  }
  if(pa < lenA){
    std::copy(a+pa,a+lenA,c+pc);
    std::copy(va+pa,va+lenA,vc+pc);
    pc += lenA - pa;
  }
  if(pb < lenB){
    std::copy(b+pb,b+lenB,c+pc);
    std::copy(vb+pb,vb+lenB,vc+pc);
    pc += lenB - pb;
  }
  return pc;
}


inline int sparse_mul(int *a,float *va, int lenA, int *b, float *vb, int lenB, int *c, float *vc){
  int pa,pb,pc;
  pa = pb = pc = 0;
  while(pa < lenA && pb < lenB){
    auto ia = a[pa];
    auto ib = b[pb];
    if(ia == ib){
        c[pc] = ia;
        vc[pc] = va[pa] * vb[pb];
        pa ++; pb ++; pc ++;
      } else if (ia < ib){
        pa++;
      } else {
        pb++;
      }
  }
  return pc;
}


