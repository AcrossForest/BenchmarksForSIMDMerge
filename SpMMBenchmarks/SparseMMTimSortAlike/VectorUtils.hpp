#pragma once
#include "SparseMatTool/format.hpp"

#if (defined __SPSP_USE_ARM__) && (defined __UseHandWriteVectorCopy__)
#include "SpSpInterface/SpSpInterface.hpp"
inline void vectorCopy(const Idx *__restrict a_pidx, const Val *__restrict a_pval,
                Size_t len, Idx *__restrict c_pidx, Val *__restrict c_pval) {
  Size_t used = 0;
  Size_t vecLen = vectorLen();

  // int vecLen = vectorLen();

  while (used < len) {
    VecBool pred_a;

    pred_a = whilelt(used, len);

    VecIdx idx_a;
    VecFlt val_a;
    idx_a = loadVec(pred_a, a_pidx + used);
    val_a = loadVec(pred_a, a_pval + used);
    storeVec(idx_a, pred_a, c_pidx + used);
    storeVec(val_a, pred_a, c_pval + used);
    used += vecLen;
  }
}

inline void vectorMoveMul(const Idx *__restrict a_pidx, const Val *__restrict a_pval,
                   Val b, Size_t len, Idx *__restrict c_pidx,
                   Val *__restrict c_pval) {
  Size_t used = 0;
  Size_t vecLen = vectorLen();

  // int vecLen = vectorLen();

  while (used < len) {
    VecBool pred_a;

    pred_a = whilelt(used, len);

    VecIdx idx_a;
    VecFlt val_a;
    idx_a = loadVec(pred_a, a_pidx + used);
    val_a = loadVec(pred_a, a_pval + used);

    val_a = vector_Mul_scalar(pred_a, val_a, b);

    storeVec(idx_a, pred_a, c_pidx + used);
    storeVec(val_a, pred_a, c_pval + used);
    used += vecLen;
  }
}

#else





inline void vectorCopy(const Idx *__restrict a_pidx, const Val *__restrict a_pval,
                Size_t len, Idx *__restrict c_pidx, Val *__restrict c_pval) {
  std::copy(a_pidx, a_pidx + len, c_pidx);
  std::copy(a_pval, a_pval + len, c_pval);
}

inline void vectorMoveMul(const Idx *__restrict a_pidx, const Val *__restrict a_pval,
                   Val b, Size_t len, Idx *__restrict c_pidx,
                   Val *__restrict c_pval) {
  std::copy(a_pidx, a_pidx + len, c_pidx);
  std::transform(a_pval, a_pval + len, c_pval,
                 [b](Val v) { return v * b; });
}

#endif