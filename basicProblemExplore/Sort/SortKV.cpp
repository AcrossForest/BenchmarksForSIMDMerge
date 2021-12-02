#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include "../SimpleBenchmarking/BenchMarking.hpp"
#include "SortAlgKV.hpp"
#include "CheckUtils.hpp"


#ifdef __SPSP_USE_ARM__
#include "bramas/sortSVE.hpp"
#include "bramas/sortSVEkv.hpp"
#endif

std::size_t by64(std::size_t sz){
  return (sz/64) * 64 + (sz%64 != 0 ?64:0);
}
std::size_t byN(std::size_t sz, uint n){
  return (sz/n) * n + (sz%n != 0 ?n:0);
}

using ElemType = int;
using ValType = int;
using PairType = std::pair<ElemType,ValType>;

int main(int argc, char **argv) {
  int baseline=1;
  int simd=1;
  // Default Vector
  int len = 10000;
  int range = 1000000;
  bool unique = false;
  for(int i=0; i<argc; ++i){
    std::string opt = argv[i];
    if (opt == "scalar") baseline=std::stoi(argv[i+1]);
    if (opt == "simd") simd=std::stoi(argv[i+1]);
    if (opt == "len") len=std::stoi(argv[i+1]);
    if (opt == "range") range=std::stoi(argv[i+1]);
    if (opt == "unique") unique=true;
  }

  std::cout << "Generated random vector length = " << len << std::endl;
  std::cout << "Milestone -4" << std::endl;

  Timmer t;
  // although we should directly use std::vector<ElemType> unsorted(by64(len));  
  // but libstdc++ 10 seems has some bug in memset when sve veclen >= 16, and array size between (1<<14)~(1<<15)
  // We have to make sure the vector length is a multiple of sve veclen...
  size_t adjustedLen = byN(len,1024);
  std::cout << "Milestone -3" << std::endl;
  std::vector<ElemType> unsorted(adjustedLen); 
  std::cout << "Milestone -2" << std::endl;
  std::vector<ValType> Vunsorted(adjustedLen); 
  std::cout << "Milestone -1" << std::endl;
  std::vector<PairType> unsortedP(adjustedLen);
  std::cout << "Milestone -0" << std::endl;

  std::vector<ElemType> s(adjustedLen); 
  std::vector<ValType> Vs(adjustedLen); 

  // Buffer
  std::vector<ElemType> b(adjustedLen); 
  std::vector<ValType> Vb(adjustedLen); 

  std::vector<ElemType> goldStandard(adjustedLen); 
  std::vector<ValType> VgoldStandard(adjustedLen);

  std::vector<ElemType> mgoldStandard(adjustedLen); 
  std::vector<ValType> VmgoldStandard(adjustedLen);

  std::vector<ElemType> spsp(adjustedLen); 
  std::vector<ValType> Vspsp(adjustedLen);

  // return -1;

  std::mt19937 gen;
  if(unique) {range = range / len;}
  std::uniform_int_distribution<ElemType> dist(0,range);
  for (int i = 0; i < len; ++i) {
    if (unique){
      unsorted[i] = dist(gen) * len + i;
    } else {
      unsorted[i] = dist(gen);
    }
    Vunsorted[i] = i;
    unsortedP[i].first = unsorted[i];
    unsortedP[i].second = Vunsorted[i];
  }

  std::vector<PairType> sP;
  std::cout << "Milestone 1" << std::endl;

  t.selfAssistMeasure("std::stable_sort (kv)",1,baseline,[&](TimmerHelper& t){
    sP = unsortedP;
    t.start();
    std::stable_sort(sP.begin(), sP.begin() + len, [](PairType a, PairType b){return a.first < b.first;});
    t.end();
  });

  for(int i=0; i<len; ++i){
      mgoldStandard[i] = sP[i].first;
      VmgoldStandard[i] = sP[i].second;
  }
  std::cout << "Milestone 2" << std::endl;

  t.selfAssistMeasure("std::sort (kv)",1,baseline,[&](TimmerHelper& t){
    sP = unsortedP;
    t.start();
    std::sort(sP.begin(), sP.begin() + len, [](PairType a, PairType b){return a.first < b.first;});
    t.end();
  });

  for(int i=0; i<len; ++i){
      goldStandard[i] = sP[i].first;
      VgoldStandard[i] = sP[i].second;
  }
  std::cout << "Milestone 3" << std::endl;
  




  using SpSpInst::CPMethod;
  std::cout << "Milestone 4" << std::endl;

  t.selfAssistMeasure("Proposed SIMD (kv)",1,simd,[&](TimmerHelper& t){
    // std::cout << "Milestone 4.5" << std::endl;
    s = unsorted;
    // std::cout << "Milestone 4.53" << std::endl;
    Vs = Vunsorted;
    // std::cout << "Milestone 4.56" << std::endl;
    t.start();
    // std::cout << "Milestone 4.59" << std::endl;
    // SortAlg::sortingKV<CPMethod::IsInt>(s.data(), Vs.data(), len);
    SortAlg::sortingKV<CPMethod::IsInt>(s.data(), b.data(), Vs.data(), Vb.data(), len);
    // std::cout << "Milestone 4.6" << std::endl;
    t.end();
    // std::cout << "Milestone 4.7" << std::endl;
  });

  std::cout << "Milestone 5" << std::endl;
  spsp = s;
  Vspsp = Vs;

#ifdef __SPSP_USE_ARM__
  t.selfAssistMeasure("Bramas SIMD (kv)",1,simd,[&](TimmerHelper& t){
    s = unsorted;
    Vs = Vunsorted;
    t.start();
    SortSVEkv::Sort<int,size_t>(s.data(), Vs.data(),len);
    t.end();
  });
#endif
  std::cout << "Milestone 6" << std::endl;


//   if(s == goldStandard && spsp == goldStandard){
//   if(s == goldStandard && Vs == VgoldStandard){
  // if(spsp == goldStandard && Vspsp == VgoldStandard){

    assertEqual("stl qs key v.s. stl ms",len, goldStandard,mgoldStandard);
    assertEqual("stl qs val v.s. stl ms (fail expected)",len, VgoldStandard,VmgoldStandard);
    assertEqual("b's key v.s. stl ms",len,mgoldStandard,s);
    assertEqual("b's val v.s. stl ms (fail expected)",len,VmgoldStandard,Vs);
    assertEqual("my key v.s. stl ms",len,mgoldStandard,spsp);
    assertEqual("my val v.s. stl ms",len,VmgoldStandard,Vspsp);
    

  t.dump();
  std::cout << "Milestone 7" << std::endl;
}