#pragma once
#include <bit>
#include "../EncodeDecode.hpp"
#include "../SpSpCommon.hpp"
#include "CPUImpl.hpp"
#include "SimEncode.hpp"
#include "UserCommon.hpp"

namespace SpSpInst {
using namespace SpSpEnum;
using namespace SpSpUserCommon;
using SpSpEncodeDecode::checkPackIdentical;
using SpSpEncodeDecode::pack;
using SpSpEncodeDecode::unpack;
using SpSpIntl::SimdCPU;
using SpSpPolicyFactory::PolicyFactory;

inline constexpr int VecLen = 8;

inline constexpr CPUInfo cpu = CPUInfo(VecLen);
inline const SimdCPU cpuImpl = SimdCPU(VecLen);

inline VecBool whilelt(const ScaIdx start, const ScaIdx end) {
  return cpuImpl.makeMask(start, end);
}

inline VecBool ptrue(){
  return cpuImpl.makeMask(0,cpu.v);
}

template <class T>
inline VReg<T> dup(T s){
  return VReg<T>(cpuImpl.v, s);
}

template <class T>
inline VReg<T> load(const VecBool& pred, const T* base) {
  return cpuImpl.load(pred, base);
}

template <class T>
inline void store(const VecBool& pred, T* base, const VReg<T>& data) {
  return cpuImpl.store(pred, base, data);
}

template <class ScaType, class IdxVec>
inline VReg<ScaType> gather(VecBool pred,const ScaType* base, IdxVec idx) {
  VReg<ScaType> out(cpuImpl.v,0);
  for(int i=0; i<cpuImpl.v; ++i){
    if(pred[i]){
      out[i] = base[idx[i]];
    }
  }
  return out;
}

template <class ScaType, class IdxType>
inline void scatter(VecBool pred, ScaType* base, VReg<IdxType> idx, VReg<ScaType> data) {
  for(int i=0; i<cpuImpl.v; ++i){
    if(pred[i]){
      base[idx[i]] = data[i];
    }
  }
}

inline int vectorLen(){
  return cpuImpl.v;
}

template <class T>
inline T lastActiveElem(const VecBool& pred, const VReg<T>& data,
                        T defaultVal) {
  return cpuImpl.lastActiveElem(pred, data, defaultVal);
}

template <class T>
inline VReg<T> simd_or(const VecBool& pred, const VReg<T>& op1,
                         const VReg<T>& op2) {
  VReg<T> out(cpuImpl.v, 0);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] | op2[i];
    }
  }
  return out;
}


inline VecBool simd_logic_not(const VecBool& pred, const VecBool& op) {
  VecBool out(cpuImpl.v, false);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = !op[i];
    }
  }
  return out;
}

inline VecBool simd_logic_or(const VecBool& pred, const VecBool& op1, const VecBool& op2) {
  VecBool out(cpuImpl.v, false);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] || op2[i];
    }
  }
  return out;
}
inline VecBool simd_logic_and(const VecBool& pred, const VecBool& op1, const VecBool& op2) {
  VecBool out(cpuImpl.v, false);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] && op2[i];
    }
  }
  return out;
}

template <class T>
inline VReg<T> simd_add(const VecBool& pred, const VReg<T>& op1,
                          const VReg<T>& op2) {
  VReg<T> out(cpuImpl.v, 0);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] + op2[i];
    }
  }
  return out;
}

template <class T>
inline VReg<T> simd_compact(const VecBool& pred, const VReg<T>& op) {
  VReg<T> out(cpuImpl.v, 0);
  int pfill = 0;
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[pfill] = op[i];
      pfill ++;
    }
  }
  return out;
}

template <class T>
inline VReg<T> simd_add_vs_merge(const VecBool& pred, const VReg<T>& op1,
                          T op2) {
  VReg<T> out = op1;
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] + op2;
    }
  }
  return out;
}
template <class T>
inline VReg<T> simd_add_vs(const VecBool& pred, const VReg<T>& op1,
                          T op2) {
  VReg<T> out(cpuImpl.v, 0);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] + op2;
    }
  }
  return out;
}

template <class T>
inline VReg<T> simd_mul_vs(const VecBool& pred, const VReg<T>& op1,
                          T op2) {
  VReg<T> out(cpuImpl.v, 0);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] * op2;
    }
  }
  return out;
}

template <class T>
inline VReg<T> simd_sub(const VecBool& pred, const VReg<T>& op1,
                          const VReg<T>& op2) {
  VReg<T> out(cpuImpl.v, 0);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] - op2[i];
    }
  }
  return out;
}

template <class T>
inline VecBool simd_cmple(const VecBool& pred, const VReg<T>& op1,
                          const VReg<T>& op2) {
  VecBool out(cpuImpl.v, false);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] <= op2[i];
    }
  }
  return out;
}
template <class T>
inline VecBool simd_cmple(const VecBool& pred, const VReg<T>& op1,
                          T op2) {
  VecBool out(cpuImpl.v, false);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] <= op2;
    }
  }
  return out;
}

template <class T>
inline VecBool simd_cmpgt(const VecBool& pred, const VReg<T>& op1,
                          const VReg<T>& op2) {
  VecBool out(cpuImpl.v, false);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] > op2[i];
    }
  }
  return out;
}
template <class T>
inline VecBool simd_cmpgt(const VecBool& pred, const VReg<T>& op1,
                          T op2) {
  VecBool out(cpuImpl.v, false);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] > op2;
    }
  }
  return out;
}
template <class T>
inline VecBool simd_cmpge(const VecBool& pred, const VReg<T>& op1,
                          const VReg<T>& op2) {
  VecBool out(cpuImpl.v, false);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] >= op2[i];
    }
  }
  return out;
}
template <class T>
inline VecBool simd_cmpge(const VecBool& pred, const VReg<T>& op1,
                          T op2) {
  VecBool out(cpuImpl.v, false);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] >= op2;
    }
  }
  return out;
}

template <class T>
inline VecBool simd_cmpeq(const VecBool& pred, const VReg<T>& op1,
                          const VReg<T>& op2) {
  VecBool out(cpuImpl.v, false);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] == op2[i];
    }
  }
  return out;
}
template <class T>
inline VecBool simd_cmpeq(const VecBool& pred, const VReg<T>& op1,
                          T op2) {
  VecBool out(cpuImpl.v, false);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] == op2;
    }
  }
  return out;
}

inline uint64_t count_active(VecBool pred, VecBool op){
  uint64_t out = 0;
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i] && op[i]) {
      out ++;
    }
  }
  return out;
}
template <class ScaType>
inline VReg<ScaType> create_index(ScaType base, ScaType step){
  VReg<ScaType> out(cpuImpl.v,0);
  for(int i=0; i<cpuImpl.v; ++i){
    out[i] = base + step * i;
  }
  return out;
}

template <class T>
inline VReg<T> simd_select(const VecBool& pred, const VReg<T>& op1,
                          const VReg<T>& op2) {
  VReg<T> out(cpuImpl.v, 0);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i];
    } else {
      out[i] = op2[i];
    }
  }
  return out;
}

template <class T>
inline VReg<T> simd_mul(const VecBool& pred, const VReg<T>& op1,
                          const VReg<T>& op2) {
  VReg<T> out(cpuImpl.v, 0);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = op1[i] * op2[i];
    }
  }
  return out;
}


inline VReg<uint32_t> simd_mul_high(const VecBool& pred, const VReg<uint32_t>& op1,
                          const VReg<uint32_t>& op2) {
  VReg<uint32_t> out(cpuImpl.v, 0);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = (uint64_t(op1[i]) *  uint64_t(op2[i])) >> 32;
    }
  }
  return out;
}

inline VReg<uint32_t> simd_mul_high_vs(const VecBool& pred, const VReg<uint32_t>& op1,
                          uint32_t op2) {
  VReg<uint32_t> out(cpuImpl.v, 0);
  for (int i = 0; i < cpuImpl.v; ++i) {
    if (pred[i]) {
      out[i] = (uint64_t(op1[i]) *  uint64_t(op2)) >> 32;
    }
  }
  return out;
}

inline VBigCmp InitBigCmp(uint64_t limit, const VecPred& predLeft,
                          const VecPred& predRight) {
  Limit limitStr = unpack<Limit>(limit);
  SpSpIntl::BigCmp bigCmp = cpuImpl.InitBigCmp(limitStr, predLeft, predRight);
  SpSpEncodeDecode::checkPackIdentical(bigCmp);
  return pack(bigCmp);
  // return cpuImpl.InitBigCmp()
}

inline VBigCmp NextBigCmpFromMatRes(uint64_t limit, const VMatRes& oldMatRes) {
  SpSpIntl::BigCmp outBigCmp = cpuImpl.NextBigCmpFromMatRes(
      unpack<Limit>(limit), unpack<SpSpIntl::MatRes>(oldMatRes));
  SpSpEncodeDecode::checkPackIdentical(outBigCmp);
  return pack(outBigCmp);
  // return pack(cpuImpl.NextBigCmpFromMatRes(
  //     unpack<Limit>(limit),
  //     unpack<SpSpIntl::MatRes>(oldMatRes)
  // ));
}

inline VBigCmp NextBigCmpFromBigCmp(uint64_t limit, const VBigCmp& oldBigCmp) {
  SpSpIntl::BigCmp outBigCmp = cpuImpl.NextBigCmpFromBigCmp(
      unpack<Limit>(limit), unpack<SpSpIntl::BigCmp>(oldBigCmp));
  SpSpEncodeDecode::checkPackIdentical(outBigCmp);
  return pack(outBigCmp);
  // return pack(cpuImpl.NextBigCmpFromBigCmp(
  //     unpack<Limit>(limit),
  //     unpack<SpSpIntl::BigCmp>(oldBigCmp)
  // ));
}

namespace HideDetail {
template <class T, CPMethod method>
inline VBigCmp KeyCombineImpl(const VBigCmp& oldBigCmp,
                              const VReg<T>& idxLeft,
                              const VReg<T>& idxRight) {
  SpSpIntl::BigCmp outBigCmp = cpuImpl.KeyCombine(
      unpack<SpSpIntl::BigCmp>(oldBigCmp), idxLeft, idxRight, method);
  SpSpEncodeDecode::checkPackIdentical(outBigCmp);
  return pack(outBigCmp);
  // return pack(cpuImpl.KeyCombine(
  //     unpack<SpSpIntl::BigCmp>(oldBigCmp),
  //     idxLeft,idxRight
  // ));
  // return pack(cpuImpl.KeyCombine(
  //     unpack<SpSpIntl::BigCmp>(oldBigCmp),
  //     idxLeft,idxRight
  // ));
}
};  // namespace HideDetail
template <CPMethod method = DefMethod<ScaIdx>>
inline VBigCmp KeyCombine(const VBigCmp& oldBigCmp,
                          const VReg<ScaIdx>& idxLeft,
                          const VReg<ScaIdx>& idxRight) {
  return HideDetail::KeyCombineImpl<ScaIdx, method>(oldBigCmp, idxLeft,
                                                    idxRight);
}
template <CPMethod method = DefMethod<ScaInt>>
inline VBigCmp KeyCombine(const VBigCmp& oldBigCmp,
                          const VReg<ScaInt>& idxLeft,
                          const VReg<ScaInt>& idxRight) {
  return HideDetail::KeyCombineImpl<ScaInt, method>(oldBigCmp, idxLeft,
                                                    idxRight);
}
template <CPMethod method = DefMethod<ScaFlt>>
inline VBigCmp KeyCombine(const VBigCmp& oldBigCmp,
                          const VReg<ScaFlt>& idxLeft,
                          const VReg<ScaFlt>& idxRight) {
  return HideDetail::KeyCombineImpl<ScaFlt, method>(oldBigCmp, idxLeft,
                                                    idxRight);
}

template <LRPart lr, class T>
inline VReg<T> BFPermute(const VBigCmp& oldBigCmp, const VReg<T>& srcLeft,
                           const VReg<T>& srcRight) {
  return cpuImpl.butterflyPermute(unpack<SpSpIntl::BigCmp>(oldBigCmp), srcLeft,
                                  srcRight, lr);
}

inline VMatRes Match(VBigCmp bigCmp, uint64_t policyMask1,
                     uint64_t policyMask2) {
  SpSpIntl::MatRes outMatRes =
      cpuImpl.Match(unpack<SpSpIntl::BigCmp>(bigCmp), policyMask1, policyMask2);
  SpSpEncodeDecode::checkPackIdentical(outMatRes);
  return pack(outMatRes);

  // return pack(cpuImpl.Match(
  //     unpack<SpSpIntl::BigCmp>(bigCmp),
  //     policyMask1,policyMask2
  // ));
}

template <SEPart part, class T>
inline VReg<T> SEPermute(VMatRes matRes, VReg<T> source,
                           uint64_t defaultAndLastVal) {
  static_assert(sizeof(T) == 4);

  SEPair<T> pair(defaultAndLastVal);

  T defaultVal = pair.def;
  T lastVal = pair.lastVal;

  return cpuImpl.SEPermute(unpack<SpSpIntl::MatRes>(matRes), source, part,
                           defaultVal, lastVal);
}

inline uint64_t GetLimit(VBigCmp bigCmp, uint64_t simPolicyMask, uint64_t op2) {
  Limit outLimit = cpuImpl.GetLimit(unpack<SpSpIntl::BigCmp>(bigCmp),
                                    simPolicyMask, unpack<GetLimitOp2>(op2));
  return pack(outLimit);

  // return pack(cpuImpl.getLimit(
  //     unpack<SpSpIntl::BigCmp>(bigCmp),
  //     simPolicyMask,
  //     unpack<GetLimitOp2>(op2)
  // ));
}

};  // namespace SpSpInst