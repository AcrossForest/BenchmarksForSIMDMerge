#pragma once
#include <stdint.h>
#include <bit>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <unordered_set>
#include "../SimpleBenchmarking/BenchMarking.hpp"
#include "SpSpInst/SpSpInterface.hpp"
#include "SortAlgUtils.hpp"

namespace SortAlg {
using namespace SpSpInst;


template <CPMethod method, Size32 K, Size32 V>
inline void shortSortKV(K *Kfrom, V *Vfrom, K *Kto, V *Vto, int len) {

  auto predLeft = whilelt(0, len);
  auto predRight = whilelt(cpu.v, len);
  auto idxLeft = load(predLeft, Kfrom);
  auto idxRight = load(predRight, Kfrom + cpu.v);
  auto valLeft = load(predLeft, Vfrom);
  auto valRight = load(predRight, Vfrom + cpu.v);


  for (int st = 0; st < cpu.logV2; ++st) {
    VBigCmp bigCmp = InitBigCmp(shortLimit + st, predLeft, predRight);
    bigCmp = KeyCombine<method>(bigCmp, idxLeft, idxRight);
    // exchange key
    auto newIdxLeft = BFPermute<LRPart::Left>(bigCmp, idxLeft, idxRight);
    auto newIdxRight = BFPermute<LRPart::Right>(bigCmp, idxLeft, idxRight);
    idxLeft = newIdxLeft;
    idxRight = newIdxRight;
    // exchange val
    auto newValLeft = BFPermute<LRPart::Left>(bigCmp, valLeft, valRight);
    auto newValRight = BFPermute<LRPart::Right>(bigCmp, valLeft, valRight);
    valLeft = newValLeft;
    valRight = newValRight;
  }

  store(predLeft, Kto, idxLeft);
  store(predRight, Kto + cpu.v, idxRight);
  store(predLeft, Vto, valLeft);
  store(predRight, Vto + cpu.v, valRight);
}


template <CPMethod method, Size32 T, Size32 V>
inline void mergeSortKV(T *AFrom, V *VAFrom, int ALen, T *BFrom, V *VBFrom , int BLen, T *to, V *Vto) {
  int pa, pb, pc;
  pa = pb = pc = 0;

  while (pa < ALen && pb < BLen) {
    auto predA = whilelt(pa, ALen);
    auto predB = whilelt(pb, BLen);
    auto idxA = load(predA, AFrom + pa);
    auto idxB = load(predB, BFrom + pb);
    auto valA = load(predA, VAFrom + pa);
    auto valB = load(predB, VBFrom + pb);
    
    VBigCmp bigCmp = InitBigCmp(longLimit, predA, predB);
    bigCmp = KeyCombine<method>(bigCmp, idxA, idxB);
    auto idxLeft = BFPermute<LRPart::Left>(bigCmp, idxA, idxB);
    auto idxRight = BFPermute<LRPart::Right>(bigCmp, idxA, idxB);
    auto valLeft = BFPermute<LRPart::Left>(bigCmp, valA, valB);
    auto valRight = BFPermute<LRPart::Right>(bigCmp, valA, valB);

    uint64_t newLimit = GetLimit(bigCmp, PolicySORT.simPolicyMask, sortOp2);
    Limit unpackLimit = unpack<Limit>(newLimit);

    auto predCLow = whilelt(0, unpackLimit.generate.A);
    store(predCLow, to + pc, idxLeft);
    store(predCLow, Vto + pc, valLeft);
    
    if (unpackLimit.generate.A > cpu.v) {
      auto predCHigh = whilelt(cpu.v, unpackLimit.generate.A);
      store(predCHigh, to + pc + cpu.v, idxRight);
      store(predCHigh, Vto + pc + cpu.v, valRight);
    }
    pa += unpackLimit.consume.A;
    pb += unpackLimit.consume.B;
    pc += unpackLimit.generate.A;
  }
  if (pa < ALen) {
    std::copy(AFrom + pa, AFrom + ALen, to + pc);
    std::copy(VAFrom + pa, VAFrom + ALen, Vto + pc);
    // safeCopy(AFrom + pa, AFrom + ALen, to + pc);
    // safeCopy(VAFrom + pa, VAFrom + ALen, Vto + pc);
    pc += ALen - pa;
  }
  if (pb < BLen) {
    std::copy(BFrom + pb, BFrom + BLen, to + pc);
    std::copy(VBFrom + pb, VBFrom + BLen, Vto + pc);
    // safeCopy(BFrom + pb, BFrom + BLen, to + pc);
    // safeCopy(VBFrom + pb, VBFrom + BLen, Vto + pc);
    pc += BLen - pb;
  }
  return;
}


template <CPMethod method, Size32 T, Size32 V>
void sortingKV(T *a, V *Va, int len) {
  MergeSortStateMachine m(len);

  std::unique_ptr<T[]> extraBuffer = std::make_unique<T[]>(len);
  std::unique_ptr<V[]> VextraBuffer = std::make_unique<V[]>(len);

  T *buf[2] = {a, extraBuffer.get()};
  V *Vbuf[2] = {Va, VextraBuffer.get()};

  while (true){
    switch (m.op){
      case CommandOp::Exit:
        goto finished;
      case CommandOp::ShortSort:
        shortSortKV<method,T>(buf[m.srcBuf] + m.start, Vbuf[m.srcBuf] + m.start, buf[m.destBuf] + m.start, Vbuf[m.destBuf] + m.start, m.total);
        break;
      case CommandOp::Merge:
        mergeSortKV<method, T>(
          buf[m.srcBuf] + m.start, Vbuf[m.srcBuf] + m.start, m.left, 
          buf[m.srcBuf] + m.start + m.left, Vbuf[m.srcBuf] + m.start + m.left , m.right,
          buf[m.destBuf] + m.start, Vbuf[m.destBuf] + m.start);
        break;
    }
    m.next();
  }
finished:
  return;
}
// I have to provide this variant due to the libstdc++ bug, we must do all allocation before we use or free any of them, otherwise causing gem5 to crash ....
template <CPMethod method, Size32 T, Size32 V>
void sortingKV(T *a, T *bufferA, V *Va, V *bufferVa, int len) {
  MergeSortStateMachine m(len);

  T *buf[2] = {a, bufferA};
  V *Vbuf[2] = {Va, bufferVa};

  while (true){
    switch (m.op){
      case CommandOp::Exit:
        goto finished;
      case CommandOp::ShortSort:
        shortSortKV<method,T>(buf[m.srcBuf] + m.start, Vbuf[m.srcBuf] + m.start, buf[m.destBuf] + m.start, Vbuf[m.destBuf] + m.start, m.total);
        break;
      case CommandOp::Merge:
        mergeSortKV<method, T>(
          buf[m.srcBuf] + m.start, Vbuf[m.srcBuf] + m.start, m.left, 
          buf[m.srcBuf] + m.start + m.left, Vbuf[m.srcBuf] + m.start + m.left , m.right,
          buf[m.destBuf] + m.start, Vbuf[m.destBuf] + m.start);
        break;
    }
    m.next();
  }
finished:
  return;
}

};  // namespace SortAlg
