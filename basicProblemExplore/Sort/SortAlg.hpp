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

template <CPMethod method, Size32 T>
inline void shortSort(T *from, T *to, int len) {

  auto predLeft = whilelt(0, len);
  auto predRight = whilelt(cpu.v, len);
  auto idxLeft = load(predLeft, from);
  auto idxRight = load(predRight, from + cpu.v);


  for (int st = 0; st < cpu.logV2; ++st) {
    VBigCmp bigCmp = InitBigCmp(shortLimit + st, predLeft, predRight);
    bigCmp = KeyCombine<method>(bigCmp, idxLeft, idxRight);
    auto newIdxLeft = BFPermute<LRPart::Left>(bigCmp, idxLeft, idxRight);
    auto newIdxRight = BFPermute<LRPart::Right>(bigCmp, idxLeft, idxRight);
    idxLeft = newIdxLeft;
    idxRight = newIdxRight;
  }

  store(predLeft, to, idxLeft);
  store(predRight, to + cpu.v, idxRight);
}

template <CPMethod method, Size32 T>
inline void mergeSort(T *AFrom, int ALen, T *BFrom, int BLen, T *to) {
  int pa, pb, pc;
  pa = pb = pc = 0;

  while (pa < ALen && pb < BLen) {
    auto predA = whilelt(pa, ALen);
    auto predB = whilelt(pb, BLen);
    auto idxA = load(predA, AFrom + pa);
    auto idxB = load(predB, BFrom + pb);
    VBigCmp bigCmp = InitBigCmp(longLimit, predA, predB);
    bigCmp = KeyCombine<method>(bigCmp, idxA, idxB);
    auto idxLeft = BFPermute<LRPart::Left>(bigCmp, idxA, idxB);
    auto idxRight = BFPermute<LRPart::Right>(bigCmp, idxA, idxB);

    uint64_t newLimit = GetLimit(bigCmp, PolicySORT.simPolicyMask, sortOp2);
    Limit unpackLimit = unpack<Limit>(newLimit);

    auto predCLow = whilelt(0, unpackLimit.generate.A);
    store(predCLow, to + pc, idxLeft);
    if (unpackLimit.generate.A > cpu.v) {
      auto predCHigh = whilelt(cpu.v, unpackLimit.generate.A);
      store(predCHigh, to + pc + cpu.v, idxRight);
    }
    pa += unpackLimit.consume.A;
    pb += unpackLimit.consume.B;
    pc += unpackLimit.generate.A;
  }
  if (pa < ALen) {
    std::copy(AFrom + pa, AFrom + ALen, to + pc);
    pc += ALen - pa;
  }
  if (pb < BLen) {
    std::copy(BFrom + pb, BFrom + BLen, to + pc);
    pc += BLen - pb;
  }
  return;
}


template <CPMethod method, Size32 T>
void sorting(T *a, int len) {
  std::unique_ptr<Task[]> ts = std::make_unique<Task[]>(64);
  std::unique_ptr<T[]> extraBuffer = std::make_unique<T[]>(len);

  T *buf[2] = {a, extraBuffer.get()};
  Task *top, *bottom;
  top = bottom = ts.get();

  bottom->srcBuf = 0;
  bottom->destBuf = 0;
  bottom->start = 0;
  bottom->len = len;
  bottom->state = SortState::Init;

  while (true) {
    if (top < bottom) {
      return;
    }
    switch (top->state) {
      case SortState::Init:
        if (top->len <= cpu.v2) {
          shortSort<method, T>(buf[top->srcBuf] + top->start,
                               buf[top->destBuf] + top->start, top->len);
          --top;
          continue;
        } else {
          auto newTop = top + 1;
          auto [left, right] = getChildLen(top->len);

          newTop->srcBuf = top->srcBuf;
          newTop->destBuf = 1 - top->destBuf;
          newTop->start = top->start;
          newTop->len = left;
          newTop->state = SortState::Init;
          top->state = SortState::LeftIssued;
          ++top;
          continue;
        }
        break;
      case SortState::LeftIssued: {
        auto newTop = top + 1;
        auto [left, right] = getChildLen(top->len);

        newTop->srcBuf = top->srcBuf;
        newTop->destBuf = 1 - top->destBuf;
        newTop->start = top->start + left;
        newTop->len = right;
        newTop->state = SortState::Init;
        top->state = SortState::RightIssued;
        ++top;
        continue;
      } break;
      case SortState::RightIssued: {
        auto [left, right] = getChildLen(top->len);
        mergeSort<method, T>(buf[1 - top->destBuf] + top->start, left,
                             buf[1 - top->destBuf] + top->start + left, right,
                             buf[top->destBuf] + top->start);
        --top;
        continue;
      } break;
      default:
        break;
    }
  }
  return;
}



template <CPMethod method, Size32 T>
void sortingStateMachine(T *a, int len) {
  MergeSortStateMachine m(len);

  std::unique_ptr<T[]> extraBuffer = std::make_unique<T[]>(len);

  T *buf[2] = {a, extraBuffer.get()};

  while (true){
    switch (m.op){
      case CommandOp::Exit:
        goto finished;
      case CommandOp::ShortSort:
        shortSort<method,T>(buf[m.srcBuf] + m.start, buf[m.destBuf] + m.start, m.total);
        break;
      case CommandOp::Merge:
        mergeSort<method, T>(
          buf[m.srcBuf] + m.start, m.left, 
          buf[m.srcBuf] + m.start + m.left, m.right,
          buf[m.destBuf] + m.start);
        break;
    }
    m.next();
  }
finished:
  return;
}
};  // namespace SortAlg
