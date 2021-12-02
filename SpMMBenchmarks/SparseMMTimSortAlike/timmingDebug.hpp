#pragma once
#include "SparseMatTool/format.hpp"
#include "M5MagicInst/m5ops.h"
// extern Size_t totalStart;
// extern Size_t totalSum;
// extern Size_t mergeStart;
// extern Size_t mergeSum;
// extern Size_t reduceStart;
// extern Size_t reduceSum;
// extern Size_t printStart;
// extern Size_t printSum;


// extern Size_t moveMulStart;
// extern Size_t moveMulSum;
// extern Size_t copyStart;
// extern Size_t copySum;

#define MoreTimmers \
  D(total) \
  D(outLoop1) \
  D(mainLoop) \
  D(loopPart1) \
  D(loopPartMovMul) \
  D(loopPartReduce) \
  D(finalReduce) \
  D(copyBack) \
  D(cleanStack) \
  D(print) \
  D(merge) \
  D(reduceMain) \
  D(reducePrepare) \
  D(reduceCopy) \
  D(reduceMisc) \
  D(analysisTimR1R2R3) \
  D(execMergeR1R2R3) \


  // D(remains) \
  // D(forloopPart1) \
  // D(stackclean) \
  // D(mainLoop) \
  // D(outLoopInit)

#define GlobalCounters \
  CNT(totalLoopCount) \
  CNT(totalProccessed) \


#ifdef __UseTimmingDebug__

#define ClearTimmer(x) x##Sum = 0;
#define StartTimmer(x) x##Start = m5_rpns();
#define RecordTimmer(x) x##Sum += m5_rpns() - x##Start;
#define DisplayTimmer(x) << "\t"#x":" << x##Sum


#define D(x) \
  extern Size_t x##Start; \
  extern Size_t x##Sum; 
MoreTimmers
#undef D



#else

/////////////////////////////////////////////
//          Stat VecEfficiency             //
/////////////////////////////////////////////

#define StartTimmer(x) 
#define RecordTimmer(x) 
#define DisplayTimmer(x) 


#endif





#ifdef __MeasureSpSpVecEfficiency__

#define CNT(x) \
  extern Size_t x;
GlobalCounters
#undef CNT

#define InitLocalCounter(x) Size_t x = 0;
#define AccumulateCounter(acc,x) acc += x;
#define IncCounter(x) x += 1;

#else

#define InitLocalCounter(x)
#define AccumulateCounter(acc,x)
#define IncCounter(x)

#endif
