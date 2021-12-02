#include "SparseMMTimSortAlike.hpp"

CSR SpMM_TimSortAlikeDirect(TimmerHelper& timmerHelper,const CSR &a, const CSR &b){
  return SpMM_TimSortAlike(timmerHelper,directVectorAdd,a,b);
}
CSR SpMM_TimSortAlikeInst(TimmerHelper& timmerHelper,const CSR &a, const CSR &b){
  return SpMM_TimSortAlike(timmerHelper,instBasedVectorAdd,a,b);
}



// inline void vectorCopy(const Idx *__restrict a_pidx, const Val *__restrict a_pval,
//                 Size_t len, Idx *__restrict c_pidx, Val *__restrict c_pval) {
//   std::copy(a_pidx, a_pidx + len, c_pidx);
//   std::copy(a_pval, a_pval + len, c_pval);
// }