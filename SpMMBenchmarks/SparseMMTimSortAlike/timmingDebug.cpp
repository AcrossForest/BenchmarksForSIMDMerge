#include "timmingDebug.hpp"

#ifdef __UseTimmingDebug__
#define D(x) \
  Size_t x##Start; \
  Size_t x##Sum; 

MoreTimmers
#undef D




#else



#endif













#ifdef __MeasureSpSpVecEfficiency__

#define CNT(x) \
  Size_t x;
GlobalCounters
#undef CNT

#else


#endif