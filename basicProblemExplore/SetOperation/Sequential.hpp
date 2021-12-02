#pragma once
#include <stdint.h>


int findUnion(uint32_t *a,int lenA, uint32_t *b, int lenB, uint32_t *c){
  int pa,pb,pc;
  pa = pb = pc = 0;
  while(pa < lenA && pb < lenB){
    auto va = a[pa];
    auto vb = b[pb];
      if(va == vb){
        c[pc++] = va;
        pa ++; pb ++;
      } else if (va < vb){
        c[pc++] = a[pa++];
      } else {
        c[pc++] = b[pb++];
      }
  }
  while(pa < lenA){
    c[pc++] = a[pa++];
  }
  while(pb < lenB){
    c[pc++] = b[pb++];
  }
  return pc;
}


int findIntersection(uint32_t *a,int lenA, uint32_t *b, int lenB, uint32_t *c){
  int pa,pb,pc;
  pa = pb = pc = 0;
  while(pa < lenA && pb < lenB){
    auto va = a[pa];
    auto vb = b[pb];
      if(va == vb){
        c[pc++] = va;
        pa ++; pb ++;
      } else if (va < vb){
        pa ++;
      } else {
        pb ++;
      }
  }
  return pc;
}

int findXOR(uint32_t *a,int lenA, uint32_t *b, int lenB, uint32_t *c){
  int pa,pb,pc;
  pa = pb = pc = 0;
  while(pa < lenA && pb < lenB){
    auto va = a[pa];
    auto vb = b[pb];
      if(va == vb){
        pa ++; pb ++;
      } else if (va < vb){
        c[pc++] = a[pa++];
      } else {
        c[pc++] = b[pb++];
      }
  }
  while(pa < lenA){
    c[pc++] = a[pa++];
  }
  while(pb < lenB){
    c[pc++] = b[pb++];
  }
  return pc;
}

int findDiff(uint32_t *a,int lenA, uint32_t *b, int lenB, uint32_t *c){
  int pa,pb,pc;
  pa = pb = pc = 0;
  while(pa < lenA && pb < lenB){
    auto va = a[pa];
    auto vb = b[pb];
      if(va == vb){
        pa ++; pb ++;
      } else if (va < vb){
        c[pc++] = a[pa++];
      } else {
        pb ++;
      }
  }
  while(pa < lenA){
    c[pc++] = a[pa++];
  }
  return pc;
}


