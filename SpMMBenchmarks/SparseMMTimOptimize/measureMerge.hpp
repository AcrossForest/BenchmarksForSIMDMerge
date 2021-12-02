#pragma once
#include "M5MagicInst/m5ops.h"
#include <stdint.h>
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

#define MergeTimmers \
  D(total) \
  D(merge) \






#ifdef __Measure_Merge__

#define MergeClearTimmer(x) MeasureMerge::clear();
#define MergeStartTimmer(x) MeasureMerge::x##Start = m5_rpns();
#define MergeRecordTimmer(x) MeasureMerge::x##Sum += m5_rpns() - MeasureMerge::x##Start;
#define MergeDisplayTimmer(x) << "\t"#x":" << MeasureMerge::x##Sum

#else //__Measure_Merge__

#define MergeClearTimmer(x)
#define MergeStartTimmer(x) 
#define MergeRecordTimmer(x) 
#define MergeDisplayTimmer(x) 


#endif //__Measure_Merge__


struct MeasureMerge{

#ifdef __Measure_Merge__
#define D(x) \
  inline static uint64_t x##Start; \
  inline static uint64_t x##Sum; 
MergeTimmers
#undef D
#endif //__Measure_Merge__

  static void clear(){
#ifdef __Measure_Merge__
#define D(x) \
  x##Start = 0; \
  x##Sum = 0; 
MergeTimmers
#undef D
#endif //__Measure_Merge__
  }

};