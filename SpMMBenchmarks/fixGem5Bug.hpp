#pragma once
#include <vector>
#include <algorithm>
inline constexpr int multiple = 64;
inline constexpr int extra = 4 * multiple;
inline std::size_t by64(std::size_t sz){
  return (sz/multiple) * multiple + (sz%multiple != 0 ?multiple:0) + extra;
}


inline std::size_t avoidGhostZone(std::size_t sz){
  return by64(std::max(sz,std::size_t(400000)));
}