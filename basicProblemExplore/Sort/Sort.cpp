#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include "../SimpleBenchmarking/BenchMarking.hpp"
#include "SortAlg.hpp"

#ifdef __SPSP_USE_ARM__
#include "bramas/sortSVE.hpp"
#endif

std::size_t by64(std::size_t sz){
  return (sz/64) * 64 + (sz%64 != 0 ?64:0);
}

using ElemType = int;

int main(int argc, char **argv) {
  int baseline=1;
  int simd=1;
  // Default Vector
  int len = 10000;
  for(int i=0; i<argc; ++i){
    std::string opt = argv[i];
    if (opt == "scalar") baseline=std::stoi(argv[i+1]);
    if (opt == "simd") simd=std::stoi(argv[i+1]);
    if (opt == "len") len=std::stoi(argv[i+1]);
  }
  
  std::cout << "Generated random vector length = " << len << std::endl;

  Timmer t;
  // although we should directly use std::vector<ElemType> unsorted(by64(len));  
  // but libstdc++ 10 seems has some bug in memset when sve veclen >= 16, and array size between (1<<14)~(1<<15)
  // We have to make sure the vector length is a multiple of sve veclen...
  std::vector<ElemType> unsorted(by64(len)); 
  std::mt19937 gen;
  std::uniform_int_distribution<ElemType> dist;
  // unsorted[0] = 3;
  for (int i = 0; i < len; ++i) {
    unsorted[i] = dist(gen);
    // unsorted[i] = unsorted[i - 1] * 16807UL % 2147483647UL;
  }

  std::vector<ElemType> s;


  // t.measure("Sort", 1, 5, [&]() {
  //   s = unsorted;
  //   std::sort(s.begin(), s.begin() + len);
  // });

  t.selfAssistMeasure("std::sort",1,baseline,[&](TimmerHelper& t){
    s = unsorted;
    t.start();
    std::sort(s.begin(), s.begin() + len);
    t.end();
  });

  // t.measure("Stable Sort", 1, 5, [&]() {
  //   s = unsorted;
  //   std::stable_sort(s.begin(), s.begin() + len );
  // });

  t.selfAssistMeasure("std::stable_sort",1,baseline,[&](TimmerHelper& t){
    s = unsorted;
    t.start();
    std::stable_sort(s.begin(), s.begin() + len );
    t.end();
  });
  


  auto goldStandard = s;

  using SpSpInst::CPMethod;
  // t.measure("SpSp Sort", 1, 5, [&]() {
  //   s = unsorted;
  //   SortAlg::sorting<CPMethod::IsUInt>(s.data(), len);
  // });

  // t.selfAssistMeasure("SpSp Sort",1,simd,[&](TimmerHelper& t){
  //   s = unsorted;
  //   t.start();
  //   SortAlg::sorting<CPMethod::IsInt>(s.data(), len);
  //   t.end();
  // });

  t.selfAssistMeasure("Proposed SIMD",1,simd,[&](TimmerHelper& t){
    s = unsorted;
    t.start();
    SortAlg::sortingStateMachine<CPMethod::IsInt>(s.data(), len);
    t.end();
  });

  auto spsp = s;

#ifdef __SPSP_USE_ARM__
  t.selfAssistMeasure("Bramas SIMD",1,simd,[&](TimmerHelper& t){
    s = unsorted;
    t.start();
    SortSVE::Sort<int,size_t>(s.data(),len);
    t.end();
  });
#endif

  // Sorting don't change the array size, but we had allocated larger vector using by64 to avoid libstdc++ bug 
  spsp.resize(len);
  s.resize(len); 
  goldStandard.resize(len);

  if(s == goldStandard && spsp == goldStandard){
    std::cout << "The answers are correct" << std::endl;
  } else {
    std::cout << "The answers are incorrect" << std::endl;
    return -1;
  }


  t.dump();
}