//////////////////////////////////////////////////////////
/// By berenger.bramas@inria.fr 2020.
/// Licence is MIT.
/// Comes without any warranty.
///
/// Code to sort an array of integer or double
/// using ARM SVE (but works only for vectors of 512 bits).
/// It also includes a partitioning function.
///
/// Please refer to the README to know how to build
/// and to have more information about the functions.
///
//////////////////////////////////////////////////////////
#ifndef SORTSVE512_HPP
#define SORTSVE512_HPP

#ifndef __ARM_FEATURE_SVE
#warning __ARM_FEATURE_SVE undefined
#endif
#include <arm_sve.h>

#include <climits>
#include <cfloat>
#include <algorithm>
#include <cstdio>

#if defined(_OPENMP)
#include <omp.h>
#include <cassert>
#else
#warning OpenMP disabled
#endif

namespace SortSVE512{


inline bool IsSorted(const svint32_t& input){
    // Inverse : 0 1 2 3 gives 3 2 1 0
    svint32_t revinput = svrev_s32(input);
    // Compare: 0 1 2 3 > 3 2 1 0 gives F F T T
    svbool_t mask = svcmpgt_s32(svptrue_b32(), input, revinput);
    // Brka : F F T T give T T F F
    svbool_t v1100 = svbrkb_b_z(svptrue_b32(), mask);
    // Inv(Brka) should be the same
    // return svcntp_b32(svptrue_b32(),svnot_b_z(svptrue_b32(),v1100)) == svcntp_b32(svptrue_b32(),mask);
    return svptest_any(svptrue_b32(),v1100) && !svptest_any(v1100,mask);
}

inline bool IsSorted(const svfloat64_t& input){
    // Inverse : 0 1 2 3 gives 3 2 1 0
    svfloat64_t revinput = svrev_f64(input);
    // Compare: 0 1 2 3 > 3 2 1 0 gives F F T T
    svbool_t mask = svcmpgt_f64(svptrue_b64(), input, revinput);
    // Brka : F F T T give T T F F
    svbool_t v1100 = svbrkb_b_z(svptrue_b64(), mask);
    // Inv(Brka) should be the same
    //return svcntp_b64(svptrue_b64(),svnot_b_z(svptrue_b64(),v1100)) == svcntp_b64(svptrue_b64(),mask);
    return svptest_any(svptrue_b64(),v1100) && !svptest_any(v1100,mask);
}

template <class IndexType>
svbool_t getTrueFalseMask32(const IndexType limite){
    //return svcmplt_s32(svptrue_b32(), svindex_s32(0, 1), svdup_s32(limite));
    return svwhilelt_b32_s32(0, limite);
}

template <class IndexType>
svbool_t getTrueFalseMask64(const IndexType limite){
    //return svcmplt_s64(svptrue_b64(), svindex_s64(0, 1), svdup_s64(limite));
    return svwhilelt_b64_s32(0, limite);
}

template <class IndexType>
svbool_t getFalseTrueMask32(const IndexType limite){
    //return svcmpge_s32(svptrue_b32(), svindex_s32(0, 1), svdup_s32(limite));
    return svnot_b_z(svptrue_b32(), svwhilelt_b32_s32(0, limite));
}

template <class IndexType>
svbool_t getFalseTrueMask64(const IndexType limite){
    //return svcmpge_s64(svptrue_b64(), svindex_s64(0, 1), svdup_s64(limite));
    return svnot_b_z(svptrue_b64(), svwhilelt_b64_s32(0, limite));
}

inline void CoreSmallSort(svfloat64_t& input){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array45670123);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
    }
}

inline void CoreSmallSort(double* __restrict__ ptr1){
    svfloat64_t input = svld1_f64(svptrue_b64(),ptr1);
    CoreSmallSort(input);
    svst1_f64(svptrue_b64(),ptr1, input);
}


inline void CoreExchangeSort2V(svfloat64_t& input, svfloat64_t& input2){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        input = svmin_f64_z(svptrue_b64(),input2, permNeigh);
        input2 = svmax_f64_z(svptrue_b64(),input2, permNeigh);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array32107654);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xF0,  permNeighMax2, permNeighMin2);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xCC,  permNeighMax2, permNeighMin2);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
    }
}

inline void CoreSmallSort2(svfloat64_t& input, svfloat64_t& input2){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array45670123);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xCC,  permNeighMax2, permNeighMin2);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xF0,  permNeighMax2, permNeighMin2);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xCC,  permNeighMax2, permNeighMin2);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
    }
    CoreExchangeSort2V(input, input2);
}

inline void CoreSmallSort2(double* __restrict__ ptr1, double* __restrict__ ptr2 ){
    svfloat64_t input1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t input2 = svld1_f64(svptrue_b64(),ptr2);
    CoreSmallSort2(input1, input2);
    svst1_f64(svptrue_b64(),ptr1, input1);
    svst1_f64(svptrue_b64(),ptr2, input2);
}


inline void CoreSmallSort3(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3 ){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    CoreSmallSort2(input, input2);
    CoreSmallSort(input3);
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh = svtbl_f64( input3, idxNoNeigh);
        input3 = svmax_f64_z(svptrue_b64(),input2, permNeigh);
        input2 = svmin_f64_z(svptrue_b64(),input2, permNeigh);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input2, inputCopy);
        input2 = svmax_f64_z(svptrue_b64(),input2, inputCopy);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array32107654);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xF0,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xF0,  permNeighMax3, permNeighMin3);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xCC,  permNeighMax3, permNeighMin3);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xAA,  permNeighMax3, permNeighMin3);
    }
}

inline void CoreSmallSort3(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3  ){
    svfloat64_t input1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t input2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t input3 = svld1_f64(svptrue_b64(),ptr3);
    CoreSmallSort3(input1, input2, input3);
    svst1_f64(svptrue_b64(),ptr1, input1);
    svst1_f64(svptrue_b64(),ptr2, input2);
    svst1_f64(svptrue_b64(),ptr3, input3);
}


inline void CoreSmallSort4(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4 ){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    CoreSmallSort2(input, input2);
    CoreSmallSort2(input3, input4);
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);

        input4 = svmax_f64_z(svptrue_b64(),input, permNeigh4);
        input = svmin_f64_z(svptrue_b64(),input, permNeigh4);

        input3 = svmax_f64_z(svptrue_b64(),input2, permNeigh3);
        input2 = svmin_f64_z(svptrue_b64(),input2, permNeigh3);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input2, inputCopy);
        input2 = svmax_f64_z(svptrue_b64(),input2, inputCopy);
    }
    {
        svfloat64_t inputCopy = input3;
        input3 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array32107654);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xF0,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xF0,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xF0,  permNeighMax4, permNeighMin4);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xCC,  permNeighMax4, permNeighMin4);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xAA,  permNeighMax4, permNeighMin4);
    }
}


inline void CoreSmallSort4(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3, double* __restrict__ ptr4  ){
    svfloat64_t input1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t input2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t input3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t input4 = svld1_f64(svptrue_b64(),ptr4);
    CoreSmallSort4(input1, input2, input3, input4);
    svst1_f64(svptrue_b64(),ptr1, input1);
    svst1_f64(svptrue_b64(),ptr2, input2);
    svst1_f64(svptrue_b64(),ptr3, input3);
    svst1_f64(svptrue_b64(),ptr4, input4);
}


inline void CoreSmallSort5(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4, svfloat64_t& input5 ){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    CoreSmallSort4(input, input2, input3, input4);
    CoreSmallSort(input5);
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);

        input5 = svmax_f64_z(svptrue_b64(),input4, permNeigh5);
        input4 = svmin_f64_z(svptrue_b64(),input4, permNeigh5);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input3, inputCopy);
        input3 = svmax_f64_z(svptrue_b64(),input3, inputCopy);
    }
    {
        svfloat64_t inputCopy = input2;
        input2 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input2, inputCopy);
        input2 = svmax_f64_z(svptrue_b64(),input2, inputCopy);
    }
    {
        svfloat64_t inputCopy = input3;
        input3 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array32107654);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xF0,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xF0,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xF0,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xF0,  permNeighMax5, permNeighMin5);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xCC,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xCC,  permNeighMax5, permNeighMin5);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xAA,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xAA,  permNeighMax5, permNeighMin5);
    }
}


inline void CoreSmallSort5(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5 ){
    svfloat64_t input1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t input2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t input3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t input4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t input5 = svld1_f64(svptrue_b64(),ptr5);
    CoreSmallSort5(input1, input2, input3, input4, input5);
    svst1_f64(svptrue_b64(),ptr1, input1);
    svst1_f64(svptrue_b64(),ptr2, input2);
    svst1_f64(svptrue_b64(),ptr3, input3);
    svst1_f64(svptrue_b64(),ptr4, input4);
    svst1_f64(svptrue_b64(),ptr5, input5);
}


inline void CoreSmallSort6(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4, svfloat64_t& input5, svfloat64_t& input6 ){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    CoreSmallSort4(input, input2, input3, input4);
    CoreSmallSort2(input5, input6);
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);

        input5 = svmax_f64_z(svptrue_b64(),input4, permNeigh5);
        input6 = svmax_f64_z(svptrue_b64(),input3, permNeigh6);

        input4 = svmin_f64_z(svptrue_b64(),input4, permNeigh5);
        input3 = svmin_f64_z(svptrue_b64(),input3, permNeigh6);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input3, inputCopy);
        input3 = svmax_f64_z(svptrue_b64(),input3, inputCopy);
    }
    {
        svfloat64_t inputCopy = input2;
        input2 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input2, inputCopy);
        input2 = svmax_f64_z(svptrue_b64(),input2, inputCopy);
    }
    {
        svfloat64_t inputCopy = input3;
        input3 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svfloat64_t inputCopy = input5;
        input5 = svmin_f64_z(svptrue_b64(),input6, inputCopy);
        input6 = svmax_f64_z(svptrue_b64(),input6, inputCopy);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array32107654);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xF0,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xF0,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xF0,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xF0,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xF0,  permNeighMax6, permNeighMin6);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xCC,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xCC,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xCC,  permNeighMax6, permNeighMin6);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xAA,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xAA,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xAA,  permNeighMax6, permNeighMin6);
    }
}


inline void CoreSmallSort6(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6 ){
    svfloat64_t input1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t input2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t input3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t input4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t input5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t input6 = svld1_f64(svptrue_b64(),ptr6);
    CoreSmallSort6(input1, input2, input3, input4, input5, input6);
    svst1_f64(svptrue_b64(),ptr1, input1);
    svst1_f64(svptrue_b64(),ptr2, input2);
    svst1_f64(svptrue_b64(),ptr3, input3);
    svst1_f64(svptrue_b64(),ptr4, input4);
    svst1_f64(svptrue_b64(),ptr5, input5);
    svst1_f64(svptrue_b64(),ptr6, input6);
}


inline void CoreSmallSort7(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                            svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7 ){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    CoreSmallSort4(input, input2, input3, input4);
    CoreSmallSort3(input5, input6, input7);
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeigh7 = svtbl_f64( input7, idxNoNeigh);

        input5 = svmax_f64_z(svptrue_b64(),input4, permNeigh5);
        input6 = svmax_f64_z(svptrue_b64(),input3, permNeigh6);
        input7 = svmax_f64_z(svptrue_b64(),input2, permNeigh7);

        input4 = svmin_f64_z(svptrue_b64(),input4, permNeigh5);
        input3 = svmin_f64_z(svptrue_b64(),input3, permNeigh6);
        input2 = svmin_f64_z(svptrue_b64(),input2, permNeigh7);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input3, inputCopy);
        input3 = svmax_f64_z(svptrue_b64(),input3, inputCopy);
    }
    {
        svfloat64_t inputCopy = input2;
        input2 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input2, inputCopy);
        input2 = svmax_f64_z(svptrue_b64(),input2, inputCopy);
    }
    {
        svfloat64_t inputCopy = input3;
        input3 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svfloat64_t inputCopy = input5;
        input5 = svmin_f64_z(svptrue_b64(),input7, inputCopy);
        input7 = svmax_f64_z(svptrue_b64(),input7, inputCopy);
    }
    {
        svfloat64_t inputCopy = input5;
        input5 = svmin_f64_z(svptrue_b64(),input6, inputCopy);
        input6 = svmax_f64_z(svptrue_b64(),input6, inputCopy);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array32107654);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeigh7 = svtbl_f64( input7, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMin7 = svmin_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax7 = svmax_f64_z(svptrue_b64(),permNeigh7, input7);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xF0,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xF0,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xF0,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xF0,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xF0,  permNeighMax6, permNeighMin6);
        input7 = svsel_f64(mask0xF0,  permNeighMax7, permNeighMin7);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeigh7 = svtbl_f64( input7, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMin7 = svmin_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax7 = svmax_f64_z(svptrue_b64(),permNeigh7, input7);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xCC,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xCC,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xCC,  permNeighMax6, permNeighMin6);
        input7 = svsel_f64(mask0xCC,  permNeighMax7, permNeighMin7);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeigh7 = svtbl_f64( input7, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMin7 = svmin_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax7 = svmax_f64_z(svptrue_b64(),permNeigh7, input7);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xAA,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xAA,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xAA,  permNeighMax6, permNeighMin6);
        input7 = svsel_f64(mask0xAA,  permNeighMax7, permNeighMin7);
    }
}


inline void CoreSmallSort7(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                            double* __restrict__ ptr7){
    svfloat64_t input1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t input2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t input3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t input4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t input5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t input6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t input7 = svld1_f64(svptrue_b64(),ptr7);
    CoreSmallSort7(input1, input2, input3, input4, input5, input6, input7);
    svst1_f64(svptrue_b64(),ptr1, input1);
    svst1_f64(svptrue_b64(),ptr2, input2);
    svst1_f64(svptrue_b64(),ptr3, input3);
    svst1_f64(svptrue_b64(),ptr4, input4);
    svst1_f64(svptrue_b64(),ptr5, input5);
    svst1_f64(svptrue_b64(),ptr6, input6);
    svst1_f64(svptrue_b64(),ptr7, input7);
}


inline void CoreSmallSort8(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                            svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8 ){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    CoreSmallSort4(input, input2, input3, input4);
    CoreSmallSort4(input5, input6, input7, input8);
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeigh7 = svtbl_f64( input7, idxNoNeigh);
        svfloat64_t permNeigh8 = svtbl_f64( input8, idxNoNeigh);

        input5 = svmax_f64_z(svptrue_b64(),input4, permNeigh5);
        input6 = svmax_f64_z(svptrue_b64(),input3, permNeigh6);
        input7 = svmax_f64_z(svptrue_b64(),input2, permNeigh7);
        input8 = svmax_f64_z(svptrue_b64(),input, permNeigh8);

        input4 = svmin_f64_z(svptrue_b64(),input4, permNeigh5);
        input3 = svmin_f64_z(svptrue_b64(),input3, permNeigh6);
        input2 = svmin_f64_z(svptrue_b64(),input2, permNeigh7);
        input = svmin_f64_z(svptrue_b64(),input, permNeigh8);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input3, inputCopy);
        input3 = svmax_f64_z(svptrue_b64(),input3, inputCopy);
    }
    {
        svfloat64_t inputCopy = input2;
        input2 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input2, inputCopy);
        input2 = svmax_f64_z(svptrue_b64(),input2, inputCopy);
    }
    {
        svfloat64_t inputCopy = input3;
        input3 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svfloat64_t inputCopy = input5;
        input5 = svmin_f64_z(svptrue_b64(),input7, inputCopy);
        input7 = svmax_f64_z(svptrue_b64(),input7, inputCopy);
    }
    {
        svfloat64_t inputCopy = input6;
        input6 = svmin_f64_z(svptrue_b64(),input8, inputCopy);
        input8 = svmax_f64_z(svptrue_b64(),input8, inputCopy);
    }
    {
        svfloat64_t inputCopy = input5;
        input5 = svmin_f64_z(svptrue_b64(),input6, inputCopy);
        input6 = svmax_f64_z(svptrue_b64(),input6, inputCopy);
    }
    {
        svfloat64_t inputCopy = input7;
        input7 = svmin_f64_z(svptrue_b64(),input8, inputCopy);
        input8 = svmax_f64_z(svptrue_b64(),input8, inputCopy);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array32107654);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeigh7 = svtbl_f64( input7, idxNoNeigh);
        svfloat64_t permNeigh8 = svtbl_f64( input8, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMin7 = svmin_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMin8 = svmin_f64_z(svptrue_b64(),permNeigh8, input8);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax7 = svmax_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMax8 = svmax_f64_z(svptrue_b64(),permNeigh8, input8);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xF0,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xF0,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xF0,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xF0,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xF0,  permNeighMax6, permNeighMin6);
        input7 = svsel_f64(mask0xF0,  permNeighMax7, permNeighMin7);
        input8 = svsel_f64(mask0xF0,  permNeighMax8, permNeighMin8);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeigh7 = svtbl_f64( input7, idxNoNeigh);
        svfloat64_t permNeigh8 = svtbl_f64( input8, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMin7 = svmin_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMin8 = svmin_f64_z(svptrue_b64(),permNeigh8, input8);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax7 = svmax_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMax8 = svmax_f64_z(svptrue_b64(),permNeigh8, input8);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xCC,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xCC,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xCC,  permNeighMax6, permNeighMin6);
        input7 = svsel_f64(mask0xCC,  permNeighMax7, permNeighMin7);
        input8 = svsel_f64(mask0xCC,  permNeighMax8, permNeighMin8);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeigh7 = svtbl_f64( input7, idxNoNeigh);
        svfloat64_t permNeigh8 = svtbl_f64( input8, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMin7 = svmin_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMin8 = svmin_f64_z(svptrue_b64(),permNeigh8, input8);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax7 = svmax_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMax8 = svmax_f64_z(svptrue_b64(),permNeigh8, input8);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xAA,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xAA,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xAA,  permNeighMax6, permNeighMin6);
        input7 = svsel_f64(mask0xAA,  permNeighMax7, permNeighMin7);
        input8 = svsel_f64(mask0xAA,  permNeighMax8, permNeighMin8);
    }
}



inline void CoreSmallSort8(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                            double* __restrict__ ptr7, double* __restrict__ ptr8 ){
    svfloat64_t input1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t input2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t input3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t input4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t input5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t input6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t input7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t input8 = svld1_f64(svptrue_b64(),ptr8);
    CoreSmallSort8(input1, input2, input3, input4, input5, input6, input7, input8);
    svst1_f64(svptrue_b64(),ptr1, input1);
    svst1_f64(svptrue_b64(),ptr2, input2);
    svst1_f64(svptrue_b64(),ptr3, input3);
    svst1_f64(svptrue_b64(),ptr4, input4);
    svst1_f64(svptrue_b64(),ptr5, input5);
    svst1_f64(svptrue_b64(),ptr6, input6);
    svst1_f64(svptrue_b64(),ptr7, input7);
    svst1_f64(svptrue_b64(),ptr8, input8);
}


inline void CoreSmallEnd1(svfloat64_t& input){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array32107654);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
    }
}

inline void CoreSmallEnd2(svfloat64_t& input, svfloat64_t& input2){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input2, inputCopy);
        input2 = svmax_f64_z(svptrue_b64(),input2, inputCopy);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array32107654);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xF0,  permNeighMax2, permNeighMin2);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xCC,  permNeighMax2, permNeighMin2);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
    }
}

inline void CoreSmallEnd3(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input3, inputCopy);
        input3 = svmax_f64_z(svptrue_b64(),input3, inputCopy);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input2, inputCopy);
        input2 = svmax_f64_z(svptrue_b64(),input2, inputCopy);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array32107654);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xF0,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xF0,  permNeighMax3, permNeighMin3);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xCC,  permNeighMax3, permNeighMin3);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xAA,  permNeighMax3, permNeighMin3);
    }
}

inline void CoreSmallEnd4(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input3, inputCopy);
        input3 = svmax_f64_z(svptrue_b64(),input3, inputCopy);
    }
    {
        svfloat64_t inputCopy = input2;
        input2 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input2, inputCopy);
        input2 = svmax_f64_z(svptrue_b64(),input2, inputCopy);
    }
    {
        svfloat64_t inputCopy = input3;
        input3 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array32107654);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xF0,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xF0,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xF0,  permNeighMax4, permNeighMin4);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xCC,  permNeighMax4, permNeighMin4);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xAA,  permNeighMax4, permNeighMin4);
    }
}

inline void CoreSmallEnd5(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                              svfloat64_t& input5){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input5, inputCopy);
        input5 = svmax_f64_z(svptrue_b64(),input5, inputCopy);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input3, inputCopy);
        input3 = svmax_f64_z(svptrue_b64(),input3, inputCopy);
    }
    {
        svfloat64_t inputCopy = input2;
        input2 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input2, inputCopy);
        input2 = svmax_f64_z(svptrue_b64(),input2, inputCopy);
    }
    {
        svfloat64_t inputCopy = input3;
        input3 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array32107654);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xF0,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xF0,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xF0,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xF0,  permNeighMax5, permNeighMin5);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xCC,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xCC,  permNeighMax5, permNeighMin5);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xAA,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xAA,  permNeighMax5, permNeighMin5);
    }
}

inline void CoreSmallEnd6(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                              svfloat64_t& input5, svfloat64_t& input6){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input5, inputCopy);
        input5 = svmax_f64_z(svptrue_b64(),input5, inputCopy);
    }
    {
        svfloat64_t inputCopy = input2;
        input2 = svmin_f64_z(svptrue_b64(),input6, inputCopy);
        input6 = svmax_f64_z(svptrue_b64(),input6, inputCopy);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input3, inputCopy);
        input3 = svmax_f64_z(svptrue_b64(),input3, inputCopy);
    }
    {
        svfloat64_t inputCopy = input2;
        input2 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input2, inputCopy);
        input2 = svmax_f64_z(svptrue_b64(),input2, inputCopy);
    }
    {
        svfloat64_t inputCopy = input3;
        input3 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svfloat64_t inputCopy = input5;
        input5 = svmin_f64_z(svptrue_b64(),input6, inputCopy);
        input6 = svmax_f64_z(svptrue_b64(),input6, inputCopy);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array32107654);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xF0,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xF0,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xF0,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xF0,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xF0,  permNeighMax6, permNeighMin6);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xCC,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xCC,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xCC,  permNeighMax6, permNeighMin6);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xAA,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xAA,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xAA,  permNeighMax6, permNeighMin6);
    }
}

inline void CoreSmallEnd7(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                              svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input5, inputCopy);
        input5 = svmax_f64_z(svptrue_b64(),input5, inputCopy);
    }
    {
        svfloat64_t inputCopy = input2;
        input2 = svmin_f64_z(svptrue_b64(),input6, inputCopy);
        input6 = svmax_f64_z(svptrue_b64(),input6, inputCopy);
    }
    {
        svfloat64_t inputCopy = input3;
        input3 = svmin_f64_z(svptrue_b64(),input7, inputCopy);
        input7 = svmax_f64_z(svptrue_b64(),input7, inputCopy);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input3, inputCopy);
        input3 = svmax_f64_z(svptrue_b64(),input3, inputCopy);
    }
    {
        svfloat64_t inputCopy = input2;
        input2 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input2, inputCopy);
        input2 = svmax_f64_z(svptrue_b64(),input2, inputCopy);
    }
    {
        svfloat64_t inputCopy = input3;
        input3 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svfloat64_t inputCopy = input5;
        input5 = svmin_f64_z(svptrue_b64(),input7, inputCopy);
        input7 = svmax_f64_z(svptrue_b64(),input7, inputCopy);
    }
    {
        svfloat64_t inputCopy = input5;
        input5 = svmin_f64_z(svptrue_b64(),input6, inputCopy);
        input6 = svmax_f64_z(svptrue_b64(),input6, inputCopy);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array32107654);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeigh7 = svtbl_f64( input7, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMin7 = svmin_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax7 = svmax_f64_z(svptrue_b64(),permNeigh7, input7);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xF0,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xF0,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xF0,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xF0,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xF0,  permNeighMax6, permNeighMin6);
        input7 = svsel_f64(mask0xF0,  permNeighMax7, permNeighMin7);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeigh7 = svtbl_f64( input7, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMin7 = svmin_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax7 = svmax_f64_z(svptrue_b64(),permNeigh7, input7);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xCC,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xCC,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xCC,  permNeighMax6, permNeighMin6);
        input7 = svsel_f64(mask0xCC,  permNeighMax7, permNeighMin7);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeigh7 = svtbl_f64( input7, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMin7 = svmin_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax7 = svmax_f64_z(svptrue_b64(),permNeigh7, input7);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xAA,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xAA,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xAA,  permNeighMax6, permNeighMin6);
        input7 = svsel_f64(mask0xAA,  permNeighMax7, permNeighMin7);
    }
}


inline void CoreSmallEnd8(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                              svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8 ){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input5, inputCopy);
        input5 = svmax_f64_z(svptrue_b64(),input5, inputCopy);
    }
    {
        svfloat64_t inputCopy = input2;
        input2 = svmin_f64_z(svptrue_b64(),input6, inputCopy);
        input6 = svmax_f64_z(svptrue_b64(),input6, inputCopy);
    }
    {
        svfloat64_t inputCopy = input3;
        input3 = svmin_f64_z(svptrue_b64(),input7, inputCopy);
        input7 = svmax_f64_z(svptrue_b64(),input7, inputCopy);
    }
    {
        svfloat64_t inputCopy = input4;
        input4 = svmin_f64_z(svptrue_b64(),input8, inputCopy);
        input8 = svmax_f64_z(svptrue_b64(),input8, inputCopy);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input3, inputCopy);
        input3 = svmax_f64_z(svptrue_b64(),input3, inputCopy);
    }
    {
        svfloat64_t inputCopy = input2;
        input2 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svfloat64_t inputCopy = input;
        input = svmin_f64_z(svptrue_b64(),input2, inputCopy);
        input2 = svmax_f64_z(svptrue_b64(),input2, inputCopy);
    }
    {
        svfloat64_t inputCopy = input3;
        input3 = svmin_f64_z(svptrue_b64(),input4, inputCopy);
        input4 = svmax_f64_z(svptrue_b64(),input4, inputCopy);
    }
    {
        svfloat64_t inputCopy = input5;
        input5 = svmin_f64_z(svptrue_b64(),input7, inputCopy);
        input7 = svmax_f64_z(svptrue_b64(),input7, inputCopy);
    }
    {
        svfloat64_t inputCopy = input6;
        input6 = svmin_f64_z(svptrue_b64(),input8, inputCopy);
        input8 = svmax_f64_z(svptrue_b64(),input8, inputCopy);
    }
    {
        svfloat64_t inputCopy = input5;
        input5 = svmin_f64_z(svptrue_b64(),input6, inputCopy);
        input6 = svmax_f64_z(svptrue_b64(),input6, inputCopy);
    }
    {
        svfloat64_t inputCopy = input7;
        input7 = svmin_f64_z(svptrue_b64(),input8, inputCopy);
        input8 = svmax_f64_z(svptrue_b64(),input8, inputCopy);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array32107654);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeigh7 = svtbl_f64( input7, idxNoNeigh);
        svfloat64_t permNeigh8 = svtbl_f64( input8, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMin7 = svmin_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMin8 = svmin_f64_z(svptrue_b64(),permNeigh8, input8);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax7 = svmax_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMax8 = svmax_f64_z(svptrue_b64(),permNeigh8, input8);
        input = svsel_f64(mask0xF0,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xF0,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xF0,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xF0,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xF0,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xF0,  permNeighMax6, permNeighMin6);
        input7 = svsel_f64(mask0xF0,  permNeighMax7, permNeighMin7);
        input8 = svsel_f64(mask0xF0,  permNeighMax8, permNeighMin8);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array54761032);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeigh7 = svtbl_f64( input7, idxNoNeigh);
        svfloat64_t permNeigh8 = svtbl_f64( input8, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMin7 = svmin_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMin8 = svmin_f64_z(svptrue_b64(),permNeigh8, input8);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax7 = svmax_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMax8 = svmax_f64_z(svptrue_b64(),permNeigh8, input8);
        input = svsel_f64(mask0xCC,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xCC,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xCC,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xCC,  permNeighMax6, permNeighMin6);
        input7 = svsel_f64(mask0xCC,  permNeighMax7, permNeighMin7);
        input8 = svsel_f64(mask0xCC,  permNeighMax8, permNeighMin8);
    }
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64(), array67452301);
        svfloat64_t permNeigh = svtbl_f64( input, idxNoNeigh);
        svfloat64_t permNeigh2 = svtbl_f64( input2, idxNoNeigh);
        svfloat64_t permNeigh3 = svtbl_f64( input3, idxNoNeigh);
        svfloat64_t permNeigh4 = svtbl_f64( input4, idxNoNeigh);
        svfloat64_t permNeigh5 = svtbl_f64( input5, idxNoNeigh);
        svfloat64_t permNeigh6 = svtbl_f64( input6, idxNoNeigh);
        svfloat64_t permNeigh7 = svtbl_f64( input7, idxNoNeigh);
        svfloat64_t permNeigh8 = svtbl_f64( input8, idxNoNeigh);
        svfloat64_t permNeighMin = svmin_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMin2 = svmin_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMin3 = svmin_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMin4 = svmin_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMin5 = svmin_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMin6 = svmin_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMin7 = svmin_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMin8 = svmin_f64_z(svptrue_b64(),permNeigh8, input8);
        svfloat64_t permNeighMax = svmax_f64_z(svptrue_b64(),permNeigh, input);
        svfloat64_t permNeighMax2 = svmax_f64_z(svptrue_b64(),permNeigh2, input2);
        svfloat64_t permNeighMax3 = svmax_f64_z(svptrue_b64(),permNeigh3, input3);
        svfloat64_t permNeighMax4 = svmax_f64_z(svptrue_b64(),permNeigh4, input4);
        svfloat64_t permNeighMax5 = svmax_f64_z(svptrue_b64(),permNeigh5, input5);
        svfloat64_t permNeighMax6 = svmax_f64_z(svptrue_b64(),permNeigh6, input6);
        svfloat64_t permNeighMax7 = svmax_f64_z(svptrue_b64(),permNeigh7, input7);
        svfloat64_t permNeighMax8 = svmax_f64_z(svptrue_b64(),permNeigh8, input8);
        input = svsel_f64(mask0xAA,  permNeighMax, permNeighMin);
        input2 = svsel_f64(mask0xAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_f64(mask0xAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_f64(mask0xAA,  permNeighMax4, permNeighMin4);
        input5 = svsel_f64(mask0xAA,  permNeighMax5, permNeighMin5);
        input6 = svsel_f64(mask0xAA,  permNeighMax6, permNeighMin6);
        input7 = svsel_f64(mask0xAA,  permNeighMax7, permNeighMin7);
        input8 = svsel_f64(mask0xAA,  permNeighMax8, permNeighMin8);
    }
}

inline void CoreSmallSort9(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                            svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8,
                            svfloat64_t& input9){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort(input9);
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh9 = svtbl_f64( input9, idxNoNeigh);

        input9 = svmax_f64_z(svptrue_b64(),input8, permNeigh9);

        input8 = svmin_f64_z(svptrue_b64(),input8, permNeigh9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd1(input9);
}



inline void CoreSmallSort9(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                            double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                            double* __restrict__ ptr7, double* __restrict__ ptr8,
                            double* __restrict__ ptr9){
    svfloat64_t input1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t input2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t input3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t input4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t input5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t input6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t input7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t input8 = svld1_f64(svptrue_b64(),ptr8);
    svfloat64_t input9 = svld1_f64(svptrue_b64(),ptr9);
    CoreSmallSort9(input1, input2, input3, input4, input5, input6, input7, input8,
                    input9);
    svst1_f64(svptrue_b64(),ptr1, input1);
    svst1_f64(svptrue_b64(),ptr2, input2);
    svst1_f64(svptrue_b64(),ptr3, input3);
    svst1_f64(svptrue_b64(),ptr4, input4);
    svst1_f64(svptrue_b64(),ptr5, input5);
    svst1_f64(svptrue_b64(),ptr6, input6);
    svst1_f64(svptrue_b64(),ptr7, input7);
    svst1_f64(svptrue_b64(),ptr8, input8);
    svst1_f64(svptrue_b64(),ptr9, input9);
}


inline void CoreSmallSort10(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                             svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8,
                             svfloat64_t& input9, svfloat64_t& input10){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort2(input9, input10);
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh9 = svtbl_f64( input9, idxNoNeigh);
        svfloat64_t permNeigh10 = svtbl_f64( input10, idxNoNeigh);

        input9 = svmax_f64_z(svptrue_b64(),input8, permNeigh9);
        input10 = svmax_f64_z(svptrue_b64(),input7, permNeigh10);

        input8 = svmin_f64_z(svptrue_b64(),input8, permNeigh9);
        input7 = svmin_f64_z(svptrue_b64(),input7, permNeigh10);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd2(input9, input10);
}



inline void CoreSmallSort10(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                             double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                             double* __restrict__ ptr7, double* __restrict__ ptr8,
                             double* __restrict__ ptr9, double* __restrict__ ptr10){
    svfloat64_t input1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t input2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t input3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t input4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t input5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t input6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t input7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t input8 = svld1_f64(svptrue_b64(),ptr8);
    svfloat64_t input9 = svld1_f64(svptrue_b64(),ptr9);
    svfloat64_t input10 = svld1_f64(svptrue_b64(),ptr10);
    CoreSmallSort10(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10);
    svst1_f64(svptrue_b64(),ptr1, input1);
    svst1_f64(svptrue_b64(),ptr2, input2);
    svst1_f64(svptrue_b64(),ptr3, input3);
    svst1_f64(svptrue_b64(),ptr4, input4);
    svst1_f64(svptrue_b64(),ptr5, input5);
    svst1_f64(svptrue_b64(),ptr6, input6);
    svst1_f64(svptrue_b64(),ptr7, input7);
    svst1_f64(svptrue_b64(),ptr8, input8);
    svst1_f64(svptrue_b64(),ptr9, input9);
    svst1_f64(svptrue_b64(),ptr10, input10);
}

inline void CoreSmallSort11(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                             svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8,
                             svfloat64_t& input9, svfloat64_t& input10, svfloat64_t& input11){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort3(input9, input10, input11);
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh9 = svtbl_f64( input9, idxNoNeigh);
        svfloat64_t permNeigh10 = svtbl_f64( input10, idxNoNeigh);
        svfloat64_t permNeigh11 = svtbl_f64( input11, idxNoNeigh);

        input9 = svmax_f64_z(svptrue_b64(),input8, permNeigh9);
        input10 = svmax_f64_z(svptrue_b64(),input7, permNeigh10);
        input11 = svmax_f64_z(svptrue_b64(),input6, permNeigh11);

        input8 = svmin_f64_z(svptrue_b64(),input8, permNeigh9);
        input7 = svmin_f64_z(svptrue_b64(),input7, permNeigh10);
        input6 = svmin_f64_z(svptrue_b64(),input6, permNeigh11);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd3(input9, input10, input11);
}



inline void CoreSmallSort11(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                             double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                             double* __restrict__ ptr7, double* __restrict__ ptr8,
                             double* __restrict__ ptr9, double* __restrict__ ptr10, double* __restrict__ ptr11){
    svfloat64_t input1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t input2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t input3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t input4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t input5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t input6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t input7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t input8 = svld1_f64(svptrue_b64(),ptr8);
    svfloat64_t input9 = svld1_f64(svptrue_b64(),ptr9);
    svfloat64_t input10 = svld1_f64(svptrue_b64(),ptr10);
    svfloat64_t input11 = svld1_f64(svptrue_b64(),ptr11);
    CoreSmallSort11(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11);
    svst1_f64(svptrue_b64(),ptr1, input1);
    svst1_f64(svptrue_b64(),ptr2, input2);
    svst1_f64(svptrue_b64(),ptr3, input3);
    svst1_f64(svptrue_b64(),ptr4, input4);
    svst1_f64(svptrue_b64(),ptr5, input5);
    svst1_f64(svptrue_b64(),ptr6, input6);
    svst1_f64(svptrue_b64(),ptr7, input7);
    svst1_f64(svptrue_b64(),ptr8, input8);
    svst1_f64(svptrue_b64(),ptr9, input9);
    svst1_f64(svptrue_b64(),ptr10, input10);
    svst1_f64(svptrue_b64(),ptr11, input11);
}

inline void CoreSmallSort12(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                             svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8,
                             svfloat64_t& input9, svfloat64_t& input10, svfloat64_t& input11, svfloat64_t& input12){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort4(input9, input10, input11, input12);
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh9 = svtbl_f64( input9, idxNoNeigh);
        svfloat64_t permNeigh10 = svtbl_f64( input10, idxNoNeigh);
        svfloat64_t permNeigh11 = svtbl_f64( input11, idxNoNeigh);
        svfloat64_t permNeigh12 = svtbl_f64( input12, idxNoNeigh);

        input9 = svmax_f64_z(svptrue_b64(),input8, permNeigh9);
        input10 = svmax_f64_z(svptrue_b64(),input7, permNeigh10);
        input11 = svmax_f64_z(svptrue_b64(),input6, permNeigh11);
        input12 = svmax_f64_z(svptrue_b64(),input5, permNeigh12);

        input8 = svmin_f64_z(svptrue_b64(),input8, permNeigh9);
        input7 = svmin_f64_z(svptrue_b64(),input7, permNeigh10);
        input6 = svmin_f64_z(svptrue_b64(),input6, permNeigh11);
        input5 = svmin_f64_z(svptrue_b64(),input5, permNeigh12);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd4(input9, input10, input11, input12);
}



inline void CoreSmallSort12(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                             double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                             double* __restrict__ ptr7, double* __restrict__ ptr8,
                             double* __restrict__ ptr9, double* __restrict__ ptr10, double* __restrict__ ptr11,
                             double* __restrict__ ptr12){
    svfloat64_t input1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t input2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t input3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t input4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t input5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t input6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t input7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t input8 = svld1_f64(svptrue_b64(),ptr8);
    svfloat64_t input9 = svld1_f64(svptrue_b64(),ptr9);
    svfloat64_t input10 = svld1_f64(svptrue_b64(),ptr10);
    svfloat64_t input11 = svld1_f64(svptrue_b64(),ptr11);
    svfloat64_t input12 = svld1_f64(svptrue_b64(),ptr12);
    CoreSmallSort12(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12);
    svst1_f64(svptrue_b64(),ptr1, input1);
    svst1_f64(svptrue_b64(),ptr2, input2);
    svst1_f64(svptrue_b64(),ptr3, input3);
    svst1_f64(svptrue_b64(),ptr4, input4);
    svst1_f64(svptrue_b64(),ptr5, input5);
    svst1_f64(svptrue_b64(),ptr6, input6);
    svst1_f64(svptrue_b64(),ptr7, input7);
    svst1_f64(svptrue_b64(),ptr8, input8);
    svst1_f64(svptrue_b64(),ptr9, input9);
    svst1_f64(svptrue_b64(),ptr10, input10);
    svst1_f64(svptrue_b64(),ptr11, input11);
    svst1_f64(svptrue_b64(),ptr12, input12);
}

inline void CoreSmallSort13(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                             svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8,
                             svfloat64_t& input9, svfloat64_t& input10, svfloat64_t& input11, svfloat64_t& input12,
                             svfloat64_t& input13){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort5(input9, input10, input11, input12, input13);
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh9 = svtbl_f64( input9, idxNoNeigh);
        svfloat64_t permNeigh10 = svtbl_f64( input10, idxNoNeigh);
        svfloat64_t permNeigh11 = svtbl_f64( input11, idxNoNeigh);
        svfloat64_t permNeigh12 = svtbl_f64( input12, idxNoNeigh);
        svfloat64_t permNeigh13 = svtbl_f64( input13, idxNoNeigh);

        input9 = svmax_f64_z(svptrue_b64(),input8, permNeigh9);
        input10 = svmax_f64_z(svptrue_b64(),input7, permNeigh10);
        input11 = svmax_f64_z(svptrue_b64(),input6, permNeigh11);
        input12 = svmax_f64_z(svptrue_b64(),input5, permNeigh12);
        input13 = svmax_f64_z(svptrue_b64(),input4, permNeigh13);

        input8 = svmin_f64_z(svptrue_b64(),input8, permNeigh9);
        input7 = svmin_f64_z(svptrue_b64(),input7, permNeigh10);
        input6 = svmin_f64_z(svptrue_b64(),input6, permNeigh11);
        input5 = svmin_f64_z(svptrue_b64(),input5, permNeigh12);
        input4 = svmin_f64_z(svptrue_b64(),input4, permNeigh13);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd5(input9, input10, input11, input12, input13);
}



inline void CoreSmallSort13(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                             double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                             double* __restrict__ ptr7, double* __restrict__ ptr8,
                             double* __restrict__ ptr9, double* __restrict__ ptr10, double* __restrict__ ptr11,
                             double* __restrict__ ptr12, double* __restrict__ ptr13){
    svfloat64_t input1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t input2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t input3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t input4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t input5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t input6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t input7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t input8 = svld1_f64(svptrue_b64(),ptr8);
    svfloat64_t input9 = svld1_f64(svptrue_b64(),ptr9);
    svfloat64_t input10 = svld1_f64(svptrue_b64(),ptr10);
    svfloat64_t input11 = svld1_f64(svptrue_b64(),ptr11);
    svfloat64_t input12 = svld1_f64(svptrue_b64(),ptr12);
    svfloat64_t input13 = svld1_f64(svptrue_b64(),ptr13);
    CoreSmallSort13(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12, input13);
    svst1_f64(svptrue_b64(),ptr1, input1);
    svst1_f64(svptrue_b64(),ptr2, input2);
    svst1_f64(svptrue_b64(),ptr3, input3);
    svst1_f64(svptrue_b64(),ptr4, input4);
    svst1_f64(svptrue_b64(),ptr5, input5);
    svst1_f64(svptrue_b64(),ptr6, input6);
    svst1_f64(svptrue_b64(),ptr7, input7);
    svst1_f64(svptrue_b64(),ptr8, input8);
    svst1_f64(svptrue_b64(),ptr9, input9);
    svst1_f64(svptrue_b64(),ptr10, input10);
    svst1_f64(svptrue_b64(),ptr11, input11);
    svst1_f64(svptrue_b64(),ptr12, input12);
    svst1_f64(svptrue_b64(),ptr13, input13);
}

inline void CoreSmallSort14(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                             svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8,
                             svfloat64_t& input9, svfloat64_t& input10, svfloat64_t& input11, svfloat64_t& input12,
                             svfloat64_t& input13, svfloat64_t& input14){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort6(input9, input10, input11, input12, input13, input14);
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh9 = svtbl_f64( input9, idxNoNeigh);
        svfloat64_t permNeigh10 = svtbl_f64( input10, idxNoNeigh);
        svfloat64_t permNeigh11 = svtbl_f64( input11, idxNoNeigh);
        svfloat64_t permNeigh12 = svtbl_f64( input12, idxNoNeigh);
        svfloat64_t permNeigh13 = svtbl_f64( input13, idxNoNeigh);
        svfloat64_t permNeigh14 = svtbl_f64( input14, idxNoNeigh);

        input9 = svmax_f64_z(svptrue_b64(),input8, permNeigh9);
        input10 = svmax_f64_z(svptrue_b64(),input7, permNeigh10);
        input11 = svmax_f64_z(svptrue_b64(),input6, permNeigh11);
        input12 = svmax_f64_z(svptrue_b64(),input5, permNeigh12);
        input13 = svmax_f64_z(svptrue_b64(),input4, permNeigh13);
        input14 = svmax_f64_z(svptrue_b64(),input3, permNeigh14);

        input8 = svmin_f64_z(svptrue_b64(),input8, permNeigh9);
        input7 = svmin_f64_z(svptrue_b64(),input7, permNeigh10);
        input6 = svmin_f64_z(svptrue_b64(),input6, permNeigh11);
        input5 = svmin_f64_z(svptrue_b64(),input5, permNeigh12);
        input4 = svmin_f64_z(svptrue_b64(),input4, permNeigh13);
        input3 = svmin_f64_z(svptrue_b64(),input3, permNeigh14);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd6(input9, input10, input11, input12, input13, input14);
}



inline void CoreSmallSort14(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                             double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                             double* __restrict__ ptr7, double* __restrict__ ptr8,
                             double* __restrict__ ptr9, double* __restrict__ ptr10, double* __restrict__ ptr11,
                             double* __restrict__ ptr12, double* __restrict__ ptr13, double* __restrict__ ptr14){
    svfloat64_t input1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t input2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t input3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t input4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t input5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t input6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t input7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t input8 = svld1_f64(svptrue_b64(),ptr8);
    svfloat64_t input9 = svld1_f64(svptrue_b64(),ptr9);
    svfloat64_t input10 = svld1_f64(svptrue_b64(),ptr10);
    svfloat64_t input11 = svld1_f64(svptrue_b64(),ptr11);
    svfloat64_t input12 = svld1_f64(svptrue_b64(),ptr12);
    svfloat64_t input13 = svld1_f64(svptrue_b64(),ptr13);
    svfloat64_t input14 = svld1_f64(svptrue_b64(),ptr14);
    CoreSmallSort14(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12, input13, input14);
    svst1_f64(svptrue_b64(),ptr1, input1);
    svst1_f64(svptrue_b64(),ptr2, input2);
    svst1_f64(svptrue_b64(),ptr3, input3);
    svst1_f64(svptrue_b64(),ptr4, input4);
    svst1_f64(svptrue_b64(),ptr5, input5);
    svst1_f64(svptrue_b64(),ptr6, input6);
    svst1_f64(svptrue_b64(),ptr7, input7);
    svst1_f64(svptrue_b64(),ptr8, input8);
    svst1_f64(svptrue_b64(),ptr9, input9);
    svst1_f64(svptrue_b64(),ptr10, input10);
    svst1_f64(svptrue_b64(),ptr11, input11);
    svst1_f64(svptrue_b64(),ptr12, input12);
    svst1_f64(svptrue_b64(),ptr13, input13);
    svst1_f64(svptrue_b64(),ptr14, input14);
}

inline void CoreSmallSort15(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                             svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8,
                             svfloat64_t& input9, svfloat64_t& input10, svfloat64_t& input11, svfloat64_t& input12,
                             svfloat64_t& input13, svfloat64_t& input14, svfloat64_t& input15){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort7(input9, input10, input11, input12, input13, input14, input15);
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh9 = svtbl_f64( input9, idxNoNeigh);
        svfloat64_t permNeigh10 = svtbl_f64( input10, idxNoNeigh);
        svfloat64_t permNeigh11 = svtbl_f64( input11, idxNoNeigh);
        svfloat64_t permNeigh12 = svtbl_f64( input12, idxNoNeigh);
        svfloat64_t permNeigh13 = svtbl_f64( input13, idxNoNeigh);
        svfloat64_t permNeigh14 = svtbl_f64( input14, idxNoNeigh);
        svfloat64_t permNeigh15 = svtbl_f64( input15, idxNoNeigh);

        input9 = svmax_f64_z(svptrue_b64(),input8, permNeigh9);
        input10 = svmax_f64_z(svptrue_b64(),input7, permNeigh10);
        input11 = svmax_f64_z(svptrue_b64(),input6, permNeigh11);
        input12 = svmax_f64_z(svptrue_b64(),input5, permNeigh12);
        input13 = svmax_f64_z(svptrue_b64(),input4, permNeigh13);
        input14 = svmax_f64_z(svptrue_b64(),input3, permNeigh14);
        input15 = svmax_f64_z(svptrue_b64(),input2, permNeigh15);

        input8 = svmin_f64_z(svptrue_b64(),input8, permNeigh9);
        input7 = svmin_f64_z(svptrue_b64(),input7, permNeigh10);
        input6 = svmin_f64_z(svptrue_b64(),input6, permNeigh11);
        input5 = svmin_f64_z(svptrue_b64(),input5, permNeigh12);
        input4 = svmin_f64_z(svptrue_b64(),input4, permNeigh13);
        input3 = svmin_f64_z(svptrue_b64(),input3, permNeigh14);
        input2 = svmin_f64_z(svptrue_b64(),input2, permNeigh15);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd7(input9, input10, input11, input12, input13, input14, input15);
}



inline void CoreSmallSort15(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                             double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                             double* __restrict__ ptr7, double* __restrict__ ptr8,
                             double* __restrict__ ptr9, double* __restrict__ ptr10, double* __restrict__ ptr11,
                             double* __restrict__ ptr12, double* __restrict__ ptr13, double* __restrict__ ptr14,
                             double* __restrict__ ptr15){
    svfloat64_t input1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t input2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t input3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t input4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t input5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t input6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t input7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t input8 = svld1_f64(svptrue_b64(),ptr8);
    svfloat64_t input9 = svld1_f64(svptrue_b64(),ptr9);
    svfloat64_t input10 = svld1_f64(svptrue_b64(),ptr10);
    svfloat64_t input11 = svld1_f64(svptrue_b64(),ptr11);
    svfloat64_t input12 = svld1_f64(svptrue_b64(),ptr12);
    svfloat64_t input13 = svld1_f64(svptrue_b64(),ptr13);
    svfloat64_t input14 = svld1_f64(svptrue_b64(),ptr14);
    svfloat64_t input15 = svld1_f64(svptrue_b64(),ptr15);
    CoreSmallSort15(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12, input13, input14, input15);
    svst1_f64(svptrue_b64(),ptr1, input1);
    svst1_f64(svptrue_b64(),ptr2, input2);
    svst1_f64(svptrue_b64(),ptr3, input3);
    svst1_f64(svptrue_b64(),ptr4, input4);
    svst1_f64(svptrue_b64(),ptr5, input5);
    svst1_f64(svptrue_b64(),ptr6, input6);
    svst1_f64(svptrue_b64(),ptr7, input7);
    svst1_f64(svptrue_b64(),ptr8, input8);
    svst1_f64(svptrue_b64(),ptr9, input9);
    svst1_f64(svptrue_b64(),ptr10, input10);
    svst1_f64(svptrue_b64(),ptr11, input11);
    svst1_f64(svptrue_b64(),ptr12, input12);
    svst1_f64(svptrue_b64(),ptr13, input13);
    svst1_f64(svptrue_b64(),ptr14, input14);
    svst1_f64(svptrue_b64(),ptr15, input15);
}


inline void CoreSmallSort16(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                             svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8,
                             svfloat64_t& input9, svfloat64_t& input10, svfloat64_t& input11, svfloat64_t& input12,
                             svfloat64_t& input13, svfloat64_t& input14, svfloat64_t& input15, svfloat64_t& input16){
    unsigned long int array01234567[8] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned long int array32107654[8] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3};
    unsigned long int array45670123[8] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4};
    unsigned long int array54761032[8] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5};
    unsigned long int array67452301[8] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6};
    svbool_t mask0xAA = svzip1_b64(svpfalse_b(),svptrue_b64());
    svbool_t mask0xCC = svzip1_b64(mask0xAA,mask0xAA);
    svbool_t mask0xF0 = svzip1_b64(mask0xCC,mask0xCC);

    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort8(input9, input10, input11, input12, input13, input14, input15, input16);
    {
        svuint64_t idxNoNeigh = svld1_u64(svptrue_b64() ,array01234567);
        svfloat64_t permNeigh9 = svtbl_f64( input9, idxNoNeigh);
        svfloat64_t permNeigh10 = svtbl_f64( input10, idxNoNeigh);
        svfloat64_t permNeigh11 = svtbl_f64( input11, idxNoNeigh);
        svfloat64_t permNeigh12 = svtbl_f64( input12, idxNoNeigh);
        svfloat64_t permNeigh13 = svtbl_f64( input13, idxNoNeigh);
        svfloat64_t permNeigh14 = svtbl_f64( input14, idxNoNeigh);
        svfloat64_t permNeigh15 = svtbl_f64( input15, idxNoNeigh);
        svfloat64_t permNeigh16 = svtbl_f64( input16, idxNoNeigh);

        input9 = svmax_f64_z(svptrue_b64(),input8, permNeigh9);
        input10 = svmax_f64_z(svptrue_b64(),input7, permNeigh10);
        input11 = svmax_f64_z(svptrue_b64(),input6, permNeigh11);
        input12 = svmax_f64_z(svptrue_b64(),input5, permNeigh12);
        input13 = svmax_f64_z(svptrue_b64(),input4, permNeigh13);
        input14 = svmax_f64_z(svptrue_b64(),input3, permNeigh14);
        input15 = svmax_f64_z(svptrue_b64(),input2, permNeigh15);
        input16 = svmax_f64_z(svptrue_b64(),input, permNeigh16);

        input8 = svmin_f64_z(svptrue_b64(),input8, permNeigh9);
        input7 = svmin_f64_z(svptrue_b64(),input7, permNeigh10);
        input6 = svmin_f64_z(svptrue_b64(),input6, permNeigh11);
        input5 = svmin_f64_z(svptrue_b64(),input5, permNeigh12);
        input4 = svmin_f64_z(svptrue_b64(),input4, permNeigh13);
        input3 = svmin_f64_z(svptrue_b64(),input3, permNeigh14);
        input2 = svmin_f64_z(svptrue_b64(),input2, permNeigh15);
        input = svmin_f64_z(svptrue_b64(),input, permNeigh16);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd8(input9, input10, input11, input12, input13, input14, input15, input16);
}



inline void CoreSmallSort16(double* __restrict__ ptr1, double* __restrict__ ptr2, double* __restrict__ ptr3,
                             double* __restrict__ ptr4, double* __restrict__ ptr5, double* __restrict__ ptr6,
                             double* __restrict__ ptr7, double* __restrict__ ptr8,
                             double* __restrict__ ptr9, double* __restrict__ ptr10, double* __restrict__ ptr11,
                             double* __restrict__ ptr12, double* __restrict__ ptr13, double* __restrict__ ptr14,
                             double* __restrict__ ptr15, double* __restrict__ ptr16){
    svfloat64_t input1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t input2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t input3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t input4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t input5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t input6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t input7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t input8 = svld1_f64(svptrue_b64(),ptr8);
    svfloat64_t input9 = svld1_f64(svptrue_b64(),ptr9);
    svfloat64_t input10 = svld1_f64(svptrue_b64(),ptr10);
    svfloat64_t input11 = svld1_f64(svptrue_b64(),ptr11);
    svfloat64_t input12 = svld1_f64(svptrue_b64(),ptr12);
    svfloat64_t input13 = svld1_f64(svptrue_b64(),ptr13);
    svfloat64_t input14 = svld1_f64(svptrue_b64(),ptr14);
    svfloat64_t input15 = svld1_f64(svptrue_b64(),ptr15);
    svfloat64_t input16 = svld1_f64(svptrue_b64(),ptr16);
    CoreSmallSort16(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12, input13, input14, input15, input16);
    svst1_f64(svptrue_b64(),ptr1, input1);
    svst1_f64(svptrue_b64(),ptr2, input2);
    svst1_f64(svptrue_b64(),ptr3, input3);
    svst1_f64(svptrue_b64(),ptr4, input4);
    svst1_f64(svptrue_b64(),ptr5, input5);
    svst1_f64(svptrue_b64(),ptr6, input6);
    svst1_f64(svptrue_b64(),ptr7, input7);
    svst1_f64(svptrue_b64(),ptr8, input8);
    svst1_f64(svptrue_b64(),ptr9, input9);
    svst1_f64(svptrue_b64(),ptr10, input10);
    svst1_f64(svptrue_b64(),ptr11, input11);
    svst1_f64(svptrue_b64(),ptr12, input12);
    svst1_f64(svptrue_b64(),ptr13, input13);
    svst1_f64(svptrue_b64(),ptr14, input14);
    svst1_f64(svptrue_b64(),ptr15, input15);
    svst1_f64(svptrue_b64(),ptr16, input16);
}


inline void CoreSmallSort(svint32_t& input){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);


    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1213141589101145670123);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array8910111213141501234567);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
    }
}

inline void CoreSmallSort(int* __restrict__ ptr1){
    svint32_t input = svld1_s32(svptrue_b32(),ptr1);
    CoreSmallSort(input);
    svst1_s32(svptrue_b32(),ptr1, input);
}


inline void CoreExchangeSort2V(svint32_t& input, svint32_t& input2 ){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);

    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        input = svmin_s32_z(svptrue_b32(),input2, permNeigh);
        input2 = svmax_s32_z(svptrue_b32(),input2, permNeigh);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
    }
}

inline void CoreSmallSort2(svint32_t& input, svint32_t& input2 ){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);

    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1213141589101145670123);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array8910111213141501234567);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
    }
    CoreExchangeSort2V(input,input2);
}

inline void CoreSmallSort2(int* __restrict__ ptr1, int* __restrict__ ptr2 ){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr2);
    CoreSmallSort2(input1, input2);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr2, input2);
}


inline void CoreSmallSort3(svint32_t& input, svint32_t& input2, svint32_t& input3 ){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);

    CoreSmallSort2(input, input2);
    CoreSmallSort(input3);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh = svtbl_s32( input3, idxNoNeigh);
        input3 = svmax_s32_z(svptrue_b32(),input2, permNeigh);
        input2 = svmin_s32_z(svptrue_b32(),input2, permNeigh);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
    }
}

inline void CoreSmallSort3(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3 ){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr3);
    CoreSmallSort3(input1, input2, input3);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr2, input2);
    svst1_s32(svptrue_b32(),ptr3, input3);
}

inline void CoreSmallSort4(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4 ){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);

    CoreSmallSort2(input, input2);
    CoreSmallSort2(input3, input4);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);

        input4 = svmax_s32_z(svptrue_b32(),input, permNeigh4);
        input = svmin_s32_z(svptrue_b32(),input, permNeigh4);

        input3 = svmax_s32_z(svptrue_b32(),input2, permNeigh3);
        input2 = svmin_s32_z(svptrue_b32(),input2, permNeigh3);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
    }
    {
        svint32_t inputCopy = input3;
        input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
    }
}

inline void CoreSmallSort4(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4 ){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr4);
    CoreSmallSort4(input1, input2, input3, input4);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr2, input2);
    svst1_s32(svptrue_b32(),ptr3, input3);
    svst1_s32(svptrue_b32(),ptr4, input4);
}


inline void CoreSmallSort5(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4, svint32_t& input5 ){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);

    CoreSmallSort4(input, input2, input3, input4);
    CoreSmallSort(input5);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);

        input5 = svmax_s32_z(svptrue_b32(),input4, permNeigh5);
        input4 = svmin_s32_z(svptrue_b32(),input4, permNeigh5);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
    }
    {
        svint32_t inputCopy = input2;
        input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
    }
    {
        svint32_t inputCopy = input3;
        input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xFF00,  permNeighMax5, permNeighMin5);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xF0F0,  permNeighMax5, permNeighMin5);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xCCCC,  permNeighMax5, permNeighMin5);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xAAAA,  permNeighMax5, permNeighMin5);
    }
}

inline void CoreSmallSort5(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4, int* __restrict__ ptr5 ){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr5);
    CoreSmallSort5(input1, input2, input3, input4, input5);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr2, input2);
    svst1_s32(svptrue_b32(),ptr3, input3);
    svst1_s32(svptrue_b32(),ptr4, input4);
    svst1_s32(svptrue_b32(),ptr5, input5);
}


inline void CoreSmallSort6(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);

    CoreSmallSort4(input, input2, input3, input4);
    CoreSmallSort2(input5, input6);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);

        input5 = svmax_s32_z(svptrue_b32(),input4, permNeigh5);
        input6 = svmax_s32_z(svptrue_b32(),input3, permNeigh6);

        input4 = svmin_s32_z(svptrue_b32(),input4, permNeigh5);
        input3 = svmin_s32_z(svptrue_b32(),input3, permNeigh6);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
    }
    {
        svint32_t inputCopy = input2;
        input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
    }
    {
        svint32_t inputCopy = input3;
        input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svint32_t inputCopy = input5;
        input5 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xFF00,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xFF00,  permNeighMax6, permNeighMin6);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xF0F0,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xF0F0,  permNeighMax6, permNeighMin6);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xCCCC,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xCCCC,  permNeighMax6, permNeighMin6);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xAAAA,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xAAAA,  permNeighMax6, permNeighMin6);
    }
}

inline void CoreSmallSort6(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
                            int* __restrict__ ptr5, int* __restrict__ ptr6 ){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr6);
    CoreSmallSort6(input1, input2, input3, input4, input5, input6);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr2, input2);
    svst1_s32(svptrue_b32(),ptr3, input3);
    svst1_s32(svptrue_b32(),ptr4, input4);
    svst1_s32(svptrue_b32(),ptr5, input5);
    svst1_s32(svptrue_b32(),ptr6, input6);
}


inline void CoreSmallSort7(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);

    CoreSmallSort4(input, input2, input3, input4);
    CoreSmallSort3(input5, input6, input7);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);

        input5 = svmax_s32_z(svptrue_b32(),input4, permNeigh5);
        input6 = svmax_s32_z(svptrue_b32(),input3, permNeigh6);
        input7 = svmax_s32_z(svptrue_b32(),input2, permNeigh7);

        input4 = svmin_s32_z(svptrue_b32(),input4, permNeigh5);
        input3 = svmin_s32_z(svptrue_b32(),input3, permNeigh6);
        input2 = svmin_s32_z(svptrue_b32(),input2, permNeigh7);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
    }
    {
        svint32_t inputCopy = input2;
        input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
    }
    {
        svint32_t inputCopy = input3;
        input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svint32_t inputCopy = input5;
        input5 = svmin_s32_z(svptrue_b32(),input7, inputCopy);
        input7 = svmax_s32_z(svptrue_b32(),input7, inputCopy);
    }
    {
        svint32_t inputCopy = input5;
        input5 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMin7 = svmin_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax7 = svmax_s32_z(svptrue_b32(),permNeigh7, input7);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xFF00,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xFF00,  permNeighMax6, permNeighMin6);
        input7 = svsel_s32(mask0xFF00,  permNeighMax7, permNeighMin7);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMin7 = svmin_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax7 = svmax_s32_z(svptrue_b32(),permNeigh7, input7);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xF0F0,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xF0F0,  permNeighMax6, permNeighMin6);
        input7 = svsel_s32(mask0xF0F0,  permNeighMax7, permNeighMin7);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMin7 = svmin_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax7 = svmax_s32_z(svptrue_b32(),permNeigh7, input7);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xCCCC,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xCCCC,  permNeighMax6, permNeighMin6);
        input7 = svsel_s32(mask0xCCCC,  permNeighMax7, permNeighMin7);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMin7 = svmin_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax7 = svmax_s32_z(svptrue_b32(),permNeigh7, input7);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xAAAA,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xAAAA,  permNeighMax6, permNeighMin6);
        input7 = svsel_s32(mask0xAAAA,  permNeighMax7, permNeighMin7);
    }
}

inline void CoreSmallSort7(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
                            int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7 ){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr7);
    CoreSmallSort7(input1, input2, input3, input4, input5, input6, input7);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr2, input2);
    svst1_s32(svptrue_b32(),ptr3, input3);
    svst1_s32(svptrue_b32(),ptr4, input4);
    svst1_s32(svptrue_b32(),ptr5, input5);
    svst1_s32(svptrue_b32(),ptr6, input6);
    svst1_s32(svptrue_b32(),ptr7, input7);
}

inline void CoreSmallSort8(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8 ){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);

    CoreSmallSort4(input, input2, input3, input4);
    CoreSmallSort4(input5, input6, input7, input8);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeigh8 = svtbl_s32( input8, idxNoNeigh);

        input5 = svmax_s32_z(svptrue_b32(),input4, permNeigh5);
        input6 = svmax_s32_z(svptrue_b32(),input3, permNeigh6);
        input7 = svmax_s32_z(svptrue_b32(),input2, permNeigh7);
        input8 = svmax_s32_z(svptrue_b32(),input, permNeigh8);

        input4 = svmin_s32_z(svptrue_b32(),input4, permNeigh5);
        input3 = svmin_s32_z(svptrue_b32(),input3, permNeigh6);
        input2 = svmin_s32_z(svptrue_b32(),input2, permNeigh7);
        input = svmin_s32_z(svptrue_b32(),input, permNeigh8);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
    }
    {
        svint32_t inputCopy = input2;
        input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
    }
    {
        svint32_t inputCopy = input3;
        input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svint32_t inputCopy = input5;
        input5 = svmin_s32_z(svptrue_b32(),input7, inputCopy);
        input7 = svmax_s32_z(svptrue_b32(),input7, inputCopy);
    }
    {
        svint32_t inputCopy = input6;
        input6 = svmin_s32_z(svptrue_b32(),input8, inputCopy);
        input8 = svmax_s32_z(svptrue_b32(),input8, inputCopy);
    }
    {
        svint32_t inputCopy = input5;
        input5 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
    }
    {
        svint32_t inputCopy = input7;
        input7 = svmin_s32_z(svptrue_b32(),input8, inputCopy);
        input8 = svmax_s32_z(svptrue_b32(),input8, inputCopy);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeigh8 = svtbl_s32( input8, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMin7 = svmin_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMin8 = svmin_s32_z(svptrue_b32(),permNeigh8, input8);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax7 = svmax_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMax8 = svmax_s32_z(svptrue_b32(),permNeigh8, input8);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xFF00,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xFF00,  permNeighMax6, permNeighMin6);
        input7 = svsel_s32(mask0xFF00,  permNeighMax7, permNeighMin7);
        input8 = svsel_s32(mask0xFF00,  permNeighMax8, permNeighMin8);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeigh8 = svtbl_s32( input8, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMin7 = svmin_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMin8 = svmin_s32_z(svptrue_b32(),permNeigh8, input8);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax7 = svmax_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMax8 = svmax_s32_z(svptrue_b32(),permNeigh8, input8);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xF0F0,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xF0F0,  permNeighMax6, permNeighMin6);
        input7 = svsel_s32(mask0xF0F0,  permNeighMax7, permNeighMin7);
        input8 = svsel_s32(mask0xF0F0,  permNeighMax8, permNeighMin8);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeigh8 = svtbl_s32( input8, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMin7 = svmin_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMin8 = svmin_s32_z(svptrue_b32(),permNeigh8, input8);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax7 = svmax_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMax8 = svmax_s32_z(svptrue_b32(),permNeigh8, input8);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xCCCC,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xCCCC,  permNeighMax6, permNeighMin6);
        input7 = svsel_s32(mask0xCCCC,  permNeighMax7, permNeighMin7);
        input8 = svsel_s32(mask0xCCCC,  permNeighMax8, permNeighMin8);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeigh8 = svtbl_s32( input8, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMin7 = svmin_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMin8 = svmin_s32_z(svptrue_b32(),permNeigh8, input8);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax7 = svmax_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMax8 = svmax_s32_z(svptrue_b32(),permNeigh8, input8);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xAAAA,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xAAAA,  permNeighMax6, permNeighMin6);
        input7 = svsel_s32(mask0xAAAA,  permNeighMax7, permNeighMin7);
        input8 = svsel_s32(mask0xAAAA,  permNeighMax8, permNeighMin8);
    }
}

inline void CoreSmallSort8(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
                            int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8 ){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr8);
    CoreSmallSort8(input1, input2, input3, input4, input5, input6, input7, input8);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr2, input2);
    svst1_s32(svptrue_b32(),ptr3, input3);
    svst1_s32(svptrue_b32(),ptr4, input4);
    svst1_s32(svptrue_b32(),ptr5, input5);
    svst1_s32(svptrue_b32(),ptr6, input6);
    svst1_s32(svptrue_b32(),ptr7, input7);
    svst1_s32(svptrue_b32(),ptr8, input8);
}

inline void CoreSmallEnd1(svint32_t& input){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);

    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
    }
}

inline void CoreSmallEnd2(svint32_t& input, svint32_t& input2){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);

    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
    }
}

inline void CoreSmallEnd3(svint32_t& input, svint32_t& input2, svint32_t& input3){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);

    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
    }
}

inline void CoreSmallEnd4(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);

    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
    }
    {
        svint32_t inputCopy = input2;
        input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
    }
    {
        svint32_t inputCopy = input3;
        input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
    }
}

inline void CoreSmallEnd5(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                              svint32_t& input5){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);

    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input5, inputCopy);
        input5 = svmax_s32_z(svptrue_b32(),input5, inputCopy);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
    }
    {
        svint32_t inputCopy = input2;
        input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
    }
    {
        svint32_t inputCopy = input3;
        input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xFF00,  permNeighMax5, permNeighMin5);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xF0F0,  permNeighMax5, permNeighMin5);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xCCCC,  permNeighMax5, permNeighMin5);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xAAAA,  permNeighMax5, permNeighMin5);
    }
}

inline void CoreSmallEnd6(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                              svint32_t& input5, svint32_t& input6){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);

    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input5, inputCopy);
        input5 = svmax_s32_z(svptrue_b32(),input5, inputCopy);
    }
    {
        svint32_t inputCopy = input2;
        input2 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
    }
    {
        svint32_t inputCopy = input2;
        input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
    }
    {
        svint32_t inputCopy = input3;
        input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svint32_t inputCopy = input5;
        input5 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xFF00,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xFF00,  permNeighMax6, permNeighMin6);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xF0F0,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xF0F0,  permNeighMax6, permNeighMin6);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xCCCC,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xCCCC,  permNeighMax6, permNeighMin6);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xAAAA,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xAAAA,  permNeighMax6, permNeighMin6);
    }
}

inline void CoreSmallEnd7(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                              svint32_t& input5, svint32_t& input6, svint32_t& input7){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);

    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input5, inputCopy);
        input5 = svmax_s32_z(svptrue_b32(),input5, inputCopy);
    }
    {
        svint32_t inputCopy = input2;
        input2 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
    }
    {
        svint32_t inputCopy = input3;
        input3 = svmin_s32_z(svptrue_b32(),input7, inputCopy);
        input7 = svmax_s32_z(svptrue_b32(),input7, inputCopy);
    }
    {
        svint32_t inputCopy = input5;
        input5 = svmin_s32_z(svptrue_b32(),input7, inputCopy);
        input7 = svmax_s32_z(svptrue_b32(),input7, inputCopy);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
    }
    {
        svint32_t inputCopy = input2;
        input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
    }
    {
        svint32_t inputCopy = input3;
        input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svint32_t inputCopy = input5;
        input5 = svmin_s32_z(svptrue_b32(),input7, inputCopy);
        input7 = svmax_s32_z(svptrue_b32(),input7, inputCopy);
    }
    {
        svint32_t inputCopy = input5;
        input5 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMin7 = svmin_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax7 = svmax_s32_z(svptrue_b32(),permNeigh7, input7);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xFF00,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xFF00,  permNeighMax6, permNeighMin6);
        input7 = svsel_s32(mask0xFF00,  permNeighMax7, permNeighMin7);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMin7 = svmin_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax7 = svmax_s32_z(svptrue_b32(),permNeigh7, input7);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xF0F0,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xF0F0,  permNeighMax6, permNeighMin6);
        input7 = svsel_s32(mask0xF0F0,  permNeighMax7, permNeighMin7);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMin7 = svmin_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax7 = svmax_s32_z(svptrue_b32(),permNeigh7, input7);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xCCCC,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xCCCC,  permNeighMax6, permNeighMin6);
        input7 = svsel_s32(mask0xCCCC,  permNeighMax7, permNeighMin7);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMin7 = svmin_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax7 = svmax_s32_z(svptrue_b32(),permNeigh7, input7);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xAAAA,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xAAAA,  permNeighMax6, permNeighMin6);
        input7 = svsel_s32(mask0xAAAA,  permNeighMax7, permNeighMin7);
    }
}

inline void CoreSmallEnd8(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                              svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8 ){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input5, inputCopy);
        input5 = svmax_s32_z(svptrue_b32(),input5, inputCopy);
    }
    {
        svint32_t inputCopy = input2;
        input2 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
    }
    {
        svint32_t inputCopy = input3;
        input3 = svmin_s32_z(svptrue_b32(),input7, inputCopy);
        input7 = svmax_s32_z(svptrue_b32(),input7, inputCopy);
    }
    {
        svint32_t inputCopy = input4;
        input4 = svmin_s32_z(svptrue_b32(),input8, inputCopy);
        input8 = svmax_s32_z(svptrue_b32(),input8, inputCopy);
    }
    {
        svint32_t inputCopy = input5;
        input5 = svmin_s32_z(svptrue_b32(),input7, inputCopy);
        input7 = svmax_s32_z(svptrue_b32(),input7, inputCopy);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
    }
    {
        svint32_t inputCopy = input2;
        input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svint32_t inputCopy = input;
        input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
    }
    {
        svint32_t inputCopy = input3;
        input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
    }
    {
        svint32_t inputCopy = input5;
        input5 = svmin_s32_z(svptrue_b32(),input7, inputCopy);
        input7 = svmax_s32_z(svptrue_b32(),input7, inputCopy);
    }
    {
        svint32_t inputCopy = input6;
        input6 = svmin_s32_z(svptrue_b32(),input8, inputCopy);
        input8 = svmax_s32_z(svptrue_b32(),input8, inputCopy);
    }
    {
        svint32_t inputCopy = input5;
        input5 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
    }
    {
        svint32_t inputCopy = input7;
        input7 = svmin_s32_z(svptrue_b32(),input8, inputCopy);
        input8 = svmax_s32_z(svptrue_b32(),input8, inputCopy);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeigh8 = svtbl_s32( input8, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMin7 = svmin_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMin8 = svmin_s32_z(svptrue_b32(),permNeigh8, input8);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax7 = svmax_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMax8 = svmax_s32_z(svptrue_b32(),permNeigh8, input8);
        input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xFF00,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xFF00,  permNeighMax6, permNeighMin6);
        input7 = svsel_s32(mask0xFF00,  permNeighMax7, permNeighMin7);
        input8 = svsel_s32(mask0xFF00,  permNeighMax8, permNeighMin8);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeigh8 = svtbl_s32( input8, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMin7 = svmin_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMin8 = svmin_s32_z(svptrue_b32(),permNeigh8, input8);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax7 = svmax_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMax8 = svmax_s32_z(svptrue_b32(),permNeigh8, input8);
        input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xF0F0,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xF0F0,  permNeighMax6, permNeighMin6);
        input7 = svsel_s32(mask0xF0F0,  permNeighMax7, permNeighMin7);
        input8 = svsel_s32(mask0xF0F0,  permNeighMax8, permNeighMin8);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeigh8 = svtbl_s32( input8, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMin7 = svmin_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMin8 = svmin_s32_z(svptrue_b32(),permNeigh8, input8);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax7 = svmax_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMax8 = svmax_s32_z(svptrue_b32(),permNeigh8, input8);
        input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xCCCC,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xCCCC,  permNeighMax6, permNeighMin6);
        input7 = svsel_s32(mask0xCCCC,  permNeighMax7, permNeighMin7);
        input8 = svsel_s32(mask0xCCCC,  permNeighMax8, permNeighMin8);
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeigh8 = svtbl_s32( input8, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMin3 = svmin_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMin4 = svmin_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMin5 = svmin_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMin6 = svmin_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMin7 = svmin_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMin8 = svmin_s32_z(svptrue_b32(),permNeigh8, input8);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax3 = svmax_s32_z(svptrue_b32(),permNeigh3, input3);
        svint32_t permNeighMax4 = svmax_s32_z(svptrue_b32(),permNeigh4, input4);
        svint32_t permNeighMax5 = svmax_s32_z(svptrue_b32(),permNeigh5, input5);
        svint32_t permNeighMax6 = svmax_s32_z(svptrue_b32(),permNeigh6, input6);
        svint32_t permNeighMax7 = svmax_s32_z(svptrue_b32(),permNeigh7, input7);
        svint32_t permNeighMax8 = svmax_s32_z(svptrue_b32(),permNeigh8, input8);
        input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
        input5 = svsel_s32(mask0xAAAA,  permNeighMax5, permNeighMin5);
        input6 = svsel_s32(mask0xAAAA,  permNeighMax6, permNeighMin6);
        input7 = svsel_s32(mask0xAAAA,  permNeighMax7, permNeighMin7);
        input8 = svsel_s32(mask0xAAAA,  permNeighMax8, permNeighMin8);
    }
}

inline void CoreSmallSort9(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    svbool_t mask0xAAAA = svzip1_b32(svpfalse_b(),svptrue_b32());
    svbool_t mask0xCCCC = svzip1_b32(mask0xAAAA,mask0xAAAA);
    svbool_t mask0xF0F0 = svzip1_b32(mask0xCCCC,mask0xCCCC);
    svbool_t mask0xFF00 = svzip1_b32(mask0xF0F0,mask0xF0F0);

    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort(input9);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh9 = svtbl_s32( input9, idxNoNeigh);

        input9 = svmax_s32_z(svptrue_b32(),input8, permNeigh9);

        input8 = svmin_s32_z(svptrue_b32(),input8, permNeigh9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd1(input9);
}

inline void CoreSmallSort9(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
                            int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8,
                            int* __restrict__ ptr9){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr8);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr9);
    CoreSmallSort9(input1, input2, input3, input4, input5, input6, input7, input8,
                    input9);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr2, input2);
    svst1_s32(svptrue_b32(),ptr3, input3);
    svst1_s32(svptrue_b32(),ptr4, input4);
    svst1_s32(svptrue_b32(),ptr5, input5);
    svst1_s32(svptrue_b32(),ptr6, input6);
    svst1_s32(svptrue_b32(),ptr7, input7);
    svst1_s32(svptrue_b32(),ptr8, input8);
    svst1_s32(svptrue_b32(),ptr9, input9);
}

inline void CoreSmallSort10(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                             svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                             svint32_t& input9, svint32_t& input10){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort2(input9, input10);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh9 = svtbl_s32( input9, idxNoNeigh);
        svint32_t permNeigh10 = svtbl_s32( input10, idxNoNeigh);

        input9 = svmax_s32_z(svptrue_b32(),input8, permNeigh9);
        input10 = svmax_s32_z(svptrue_b32(),input7, permNeigh10);

        input8 = svmin_s32_z(svptrue_b32(),input8, permNeigh9);
        input7 = svmin_s32_z(svptrue_b32(),input7, permNeigh10);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd2(input9, input10);
}

inline void CoreSmallSort10(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
                             int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8,
                             int* __restrict__ ptr9, int* __restrict__ ptr10){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr8);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr9);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr10);
    CoreSmallSort10(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr2, input2);
    svst1_s32(svptrue_b32(),ptr3, input3);
    svst1_s32(svptrue_b32(),ptr4, input4);
    svst1_s32(svptrue_b32(),ptr5, input5);
    svst1_s32(svptrue_b32(),ptr6, input6);
    svst1_s32(svptrue_b32(),ptr7, input7);
    svst1_s32(svptrue_b32(),ptr8, input8);
    svst1_s32(svptrue_b32(),ptr9, input9);
    svst1_s32(svptrue_b32(),ptr10, input10);
}

inline void CoreSmallSort11(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                             svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                             svint32_t& input9, svint32_t& input10, svint32_t& input11){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort3(input9, input10, input11);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh9 = svtbl_s32( input9, idxNoNeigh);
        svint32_t permNeigh10 = svtbl_s32( input10, idxNoNeigh);
        svint32_t permNeigh11 = svtbl_s32( input11, idxNoNeigh);

        input9 = svmax_s32_z(svptrue_b32(),input8, permNeigh9);
        input10 = svmax_s32_z(svptrue_b32(),input7, permNeigh10);
        input11 = svmax_s32_z(svptrue_b32(),input6, permNeigh11);

        input8 = svmin_s32_z(svptrue_b32(),input8, permNeigh9);
        input7 = svmin_s32_z(svptrue_b32(),input7, permNeigh10);
        input6 = svmin_s32_z(svptrue_b32(),input6, permNeigh11);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd3(input9, input10, input11);
}

inline void CoreSmallSort11(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
                             int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8,
                             int* __restrict__ ptr9, int* __restrict__ ptr10, int* __restrict__ ptr11){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr8);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr9);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr10);
    svint32_t input11 = svld1_s32(svptrue_b32(),ptr11);
    CoreSmallSort11(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr2, input2);
    svst1_s32(svptrue_b32(),ptr3, input3);
    svst1_s32(svptrue_b32(),ptr4, input4);
    svst1_s32(svptrue_b32(),ptr5, input5);
    svst1_s32(svptrue_b32(),ptr6, input6);
    svst1_s32(svptrue_b32(),ptr7, input7);
    svst1_s32(svptrue_b32(),ptr8, input8);
    svst1_s32(svptrue_b32(),ptr9, input9);
    svst1_s32(svptrue_b32(),ptr10, input10);
    svst1_s32(svptrue_b32(),ptr11, input11);
}

inline void CoreSmallSort12(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                             svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                             svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12 ){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort4(input9, input10, input11, input12);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh9 = svtbl_s32( input9, idxNoNeigh);
        svint32_t permNeigh10 = svtbl_s32( input10, idxNoNeigh);
        svint32_t permNeigh11 = svtbl_s32( input11, idxNoNeigh);
        svint32_t permNeigh12 = svtbl_s32( input12, idxNoNeigh);

        input9 = svmax_s32_z(svptrue_b32(),input8, permNeigh9);
        input10 = svmax_s32_z(svptrue_b32(),input7, permNeigh10);
        input11 = svmax_s32_z(svptrue_b32(),input6, permNeigh11);
        input12 = svmax_s32_z(svptrue_b32(),input5, permNeigh12);

        input8 = svmin_s32_z(svptrue_b32(),input8, permNeigh9);
        input7 = svmin_s32_z(svptrue_b32(),input7, permNeigh10);
        input6 = svmin_s32_z(svptrue_b32(),input6, permNeigh11);
        input5 = svmin_s32_z(svptrue_b32(),input5, permNeigh12);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd4(input9, input10, input11, input12);
}

inline void CoreSmallSort12(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
                             int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8,
                             int* __restrict__ ptr9, int* __restrict__ ptr10, int* __restrict__ ptr11, int* __restrict__ ptr12){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr8);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr9);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr10);
    svint32_t input11 = svld1_s32(svptrue_b32(),ptr11);
    svint32_t input12 = svld1_s32(svptrue_b32(),ptr12);
    CoreSmallSort12(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr2, input2);
    svst1_s32(svptrue_b32(),ptr3, input3);
    svst1_s32(svptrue_b32(),ptr4, input4);
    svst1_s32(svptrue_b32(),ptr5, input5);
    svst1_s32(svptrue_b32(),ptr6, input6);
    svst1_s32(svptrue_b32(),ptr7, input7);
    svst1_s32(svptrue_b32(),ptr8, input8);
    svst1_s32(svptrue_b32(),ptr9, input9);
    svst1_s32(svptrue_b32(),ptr10, input10);
    svst1_s32(svptrue_b32(),ptr11, input11);
    svst1_s32(svptrue_b32(),ptr12, input12);
}

inline void CoreSmallSort13(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                             svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                             svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                             svint32_t& input13 ){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort5(input9, input10, input11, input12, input13);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh9 = svtbl_s32( input9, idxNoNeigh);
        svint32_t permNeigh10 = svtbl_s32( input10, idxNoNeigh);
        svint32_t permNeigh11 = svtbl_s32( input11, idxNoNeigh);
        svint32_t permNeigh12 = svtbl_s32( input12, idxNoNeigh);
        svint32_t permNeigh13 = svtbl_s32( input13, idxNoNeigh);

        input9 = svmax_s32_z(svptrue_b32(),input8, permNeigh9);
        input10 = svmax_s32_z(svptrue_b32(),input7, permNeigh10);
        input11 = svmax_s32_z(svptrue_b32(),input6, permNeigh11);
        input12 = svmax_s32_z(svptrue_b32(),input5, permNeigh12);
        input13 = svmax_s32_z(svptrue_b32(),input4, permNeigh13);

        input8 = svmin_s32_z(svptrue_b32(),input8, permNeigh9);
        input7 = svmin_s32_z(svptrue_b32(),input7, permNeigh10);
        input6 = svmin_s32_z(svptrue_b32(),input6, permNeigh11);
        input5 = svmin_s32_z(svptrue_b32(),input5, permNeigh12);
        input4 = svmin_s32_z(svptrue_b32(),input4, permNeigh13);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd5(input9, input10, input11, input12, input13);
}

inline void CoreSmallSort13(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
                             int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8,
                             int* __restrict__ ptr9, int* __restrict__ ptr10, int* __restrict__ ptr11, int* __restrict__ ptr12,
                             int* __restrict__ ptr13){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr8);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr9);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr10);
    svint32_t input11 = svld1_s32(svptrue_b32(),ptr11);
    svint32_t input12 = svld1_s32(svptrue_b32(),ptr12);
    svint32_t input13 = svld1_s32(svptrue_b32(),ptr13);
    CoreSmallSort13(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12, input13);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr2, input2);
    svst1_s32(svptrue_b32(),ptr3, input3);
    svst1_s32(svptrue_b32(),ptr4, input4);
    svst1_s32(svptrue_b32(),ptr5, input5);
    svst1_s32(svptrue_b32(),ptr6, input6);
    svst1_s32(svptrue_b32(),ptr7, input7);
    svst1_s32(svptrue_b32(),ptr8, input8);
    svst1_s32(svptrue_b32(),ptr9, input9);
    svst1_s32(svptrue_b32(),ptr10, input10);
    svst1_s32(svptrue_b32(),ptr11, input11);
    svst1_s32(svptrue_b32(),ptr12, input12);
    svst1_s32(svptrue_b32(),ptr13, input13);
}


inline void CoreSmallSort14(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                             svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                             svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                             svint32_t& input13, svint32_t& input14 ){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort6(input9, input10, input11, input12, input13, input14);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh9 = svtbl_s32( input9, idxNoNeigh);
        svint32_t permNeigh10 = svtbl_s32( input10, idxNoNeigh);
        svint32_t permNeigh11 = svtbl_s32( input11, idxNoNeigh);
        svint32_t permNeigh12 = svtbl_s32( input12, idxNoNeigh);
        svint32_t permNeigh13 = svtbl_s32( input13, idxNoNeigh);
        svint32_t permNeigh14 = svtbl_s32( input14, idxNoNeigh);

        input9 = svmax_s32_z(svptrue_b32(),input8, permNeigh9);
        input10 = svmax_s32_z(svptrue_b32(),input7, permNeigh10);
        input11 = svmax_s32_z(svptrue_b32(),input6, permNeigh11);
        input12 = svmax_s32_z(svptrue_b32(),input5, permNeigh12);
        input13 = svmax_s32_z(svptrue_b32(),input4, permNeigh13);
        input14 = svmax_s32_z(svptrue_b32(),input3, permNeigh14);

        input8 = svmin_s32_z(svptrue_b32(),input8, permNeigh9);
        input7 = svmin_s32_z(svptrue_b32(),input7, permNeigh10);
        input6 = svmin_s32_z(svptrue_b32(),input6, permNeigh11);
        input5 = svmin_s32_z(svptrue_b32(),input5, permNeigh12);
        input4 = svmin_s32_z(svptrue_b32(),input4, permNeigh13);
        input3 = svmin_s32_z(svptrue_b32(),input3, permNeigh14);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd6(input9, input10, input11, input12, input13, input14);
}

inline void CoreSmallSort14(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
                             int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8,
                             int* __restrict__ ptr9, int* __restrict__ ptr10, int* __restrict__ ptr11, int* __restrict__ ptr12,
                             int* __restrict__ ptr13, int* __restrict__ ptr14){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr8);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr9);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr10);
    svint32_t input11 = svld1_s32(svptrue_b32(),ptr11);
    svint32_t input12 = svld1_s32(svptrue_b32(),ptr12);
    svint32_t input13 = svld1_s32(svptrue_b32(),ptr13);
    svint32_t input14 = svld1_s32(svptrue_b32(),ptr14);
    CoreSmallSort14(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12, input13, input14);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr2, input2);
    svst1_s32(svptrue_b32(),ptr3, input3);
    svst1_s32(svptrue_b32(),ptr4, input4);
    svst1_s32(svptrue_b32(),ptr5, input5);
    svst1_s32(svptrue_b32(),ptr6, input6);
    svst1_s32(svptrue_b32(),ptr7, input7);
    svst1_s32(svptrue_b32(),ptr8, input8);
    svst1_s32(svptrue_b32(),ptr9, input9);
    svst1_s32(svptrue_b32(),ptr10, input10);
    svst1_s32(svptrue_b32(),ptr11, input11);
    svst1_s32(svptrue_b32(),ptr12, input12);
    svst1_s32(svptrue_b32(),ptr13, input13);
    svst1_s32(svptrue_b32(),ptr14, input14);
}


inline void CoreSmallSort15(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                             svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                             svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                             svint32_t& input13, svint32_t& input14, svint32_t& input15 ){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort7(input9, input10, input11, input12, input13, input14, input15);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh9 = svtbl_s32( input9, idxNoNeigh);
        svint32_t permNeigh10 = svtbl_s32( input10, idxNoNeigh);
        svint32_t permNeigh11 = svtbl_s32( input11, idxNoNeigh);
        svint32_t permNeigh12 = svtbl_s32( input12, idxNoNeigh);
        svint32_t permNeigh13 = svtbl_s32( input13, idxNoNeigh);
        svint32_t permNeigh14 = svtbl_s32( input14, idxNoNeigh);
        svint32_t permNeigh15 = svtbl_s32( input15, idxNoNeigh);

        input9 = svmax_s32_z(svptrue_b32(),input8, permNeigh9);
        input10 = svmax_s32_z(svptrue_b32(),input7, permNeigh10);
        input11 = svmax_s32_z(svptrue_b32(),input6, permNeigh11);
        input12 = svmax_s32_z(svptrue_b32(),input5, permNeigh12);
        input13 = svmax_s32_z(svptrue_b32(),input4, permNeigh13);
        input14 = svmax_s32_z(svptrue_b32(),input3, permNeigh14);
        input15 = svmax_s32_z(svptrue_b32(),input2, permNeigh15);

        input8 = svmin_s32_z(svptrue_b32(),input8, permNeigh9);
        input7 = svmin_s32_z(svptrue_b32(),input7, permNeigh10);
        input6 = svmin_s32_z(svptrue_b32(),input6, permNeigh11);
        input5 = svmin_s32_z(svptrue_b32(),input5, permNeigh12);
        input4 = svmin_s32_z(svptrue_b32(),input4, permNeigh13);
        input3 = svmin_s32_z(svptrue_b32(),input3, permNeigh14);
        input2 = svmin_s32_z(svptrue_b32(),input2, permNeigh15);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd7(input9, input10, input11, input12, input13, input14, input15);
}

inline void CoreSmallSort15(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
                             int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8,
                             int* __restrict__ ptr9, int* __restrict__ ptr10, int* __restrict__ ptr11, int* __restrict__ ptr12,
                             int* __restrict__ ptr13, int* __restrict__ ptr14, int* __restrict__ ptr15){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr8);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr9);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr10);
    svint32_t input11 = svld1_s32(svptrue_b32(),ptr11);
    svint32_t input12 = svld1_s32(svptrue_b32(),ptr12);
    svint32_t input13 = svld1_s32(svptrue_b32(),ptr13);
    svint32_t input14 = svld1_s32(svptrue_b32(),ptr14);
    svint32_t input15 = svld1_s32(svptrue_b32(),ptr15);
    CoreSmallSort15(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12, input13, input14, input15);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr2, input2);
    svst1_s32(svptrue_b32(),ptr3, input3);
    svst1_s32(svptrue_b32(),ptr4, input4);
    svst1_s32(svptrue_b32(),ptr5, input5);
    svst1_s32(svptrue_b32(),ptr6, input6);
    svst1_s32(svptrue_b32(),ptr7, input7);
    svst1_s32(svptrue_b32(),ptr8, input8);
    svst1_s32(svptrue_b32(),ptr9, input9);
    svst1_s32(svptrue_b32(),ptr10, input10);
    svst1_s32(svptrue_b32(),ptr11, input11);
    svst1_s32(svptrue_b32(),ptr12, input12);
    svst1_s32(svptrue_b32(),ptr13, input13);
    svst1_s32(svptrue_b32(),ptr14, input14);
    svst1_s32(svptrue_b32(),ptr15, input15);
}


inline void CoreSmallSort16(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                             svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                             svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                             svint32_t& input13, svint32_t& input14, svint32_t& input15, svint32_t& input16 ){
    unsigned int array0123456789101112131415[16] = {15 ,14 ,13 ,12 ,11 ,10 ,9 ,8  ,7 ,6 ,5 ,4 ,3 ,2 ,1 ,0};
    unsigned int array1110981514131232107654[16] = {4 ,5 ,6 ,7 ,0 ,1 ,2 ,3   ,12 ,13 ,14 ,15 ,8 ,9 ,10 ,11 };
    unsigned int array1213141589101145670123[16] = {3 ,2 ,1 ,0 ,7 ,6 ,5 ,4  ,11 ,10 ,9 ,8 ,15 ,14 ,13 ,12};
    unsigned int array1312151498111054761032[16] = {2 ,3 ,0 ,1 ,6 ,7 ,4 ,5  ,10 ,11 ,8 ,9 ,14 ,15 ,12 ,13};
    unsigned int array1415121310118967452301[16] = {1 ,0 ,3 ,2 ,5 ,4 ,7 ,6  ,9 ,8 ,11 ,10 ,13 ,12 ,15 ,14};
    unsigned int array7654321015141312111098[16] = {8 ,9 ,10 ,11 ,12 ,13 ,14 ,15  ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
    unsigned int array8910111213141501234567[16] = {7 ,6 ,5 ,4 ,3 ,2 ,1 ,0  ,15 ,14 ,13 ,12 ,11 ,10 ,9 ,8};
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort8(input9, input10, input11, input12, input13, input14, input15, input16);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32() ,array0123456789101112131415);
        svint32_t permNeigh9 = svtbl_s32( input9, idxNoNeigh);
        svint32_t permNeigh10 = svtbl_s32( input10, idxNoNeigh);
        svint32_t permNeigh11 = svtbl_s32( input11, idxNoNeigh);
        svint32_t permNeigh12 = svtbl_s32( input12, idxNoNeigh);
        svint32_t permNeigh13 = svtbl_s32( input13, idxNoNeigh);
        svint32_t permNeigh14 = svtbl_s32( input14, idxNoNeigh);
        svint32_t permNeigh15 = svtbl_s32( input15, idxNoNeigh);
        svint32_t permNeigh16 = svtbl_s32( input16, idxNoNeigh);

        input9 = svmax_s32_z(svptrue_b32(),input8, permNeigh9);
        input10 = svmax_s32_z(svptrue_b32(),input7, permNeigh10);
        input11 = svmax_s32_z(svptrue_b32(),input6, permNeigh11);
        input12 = svmax_s32_z(svptrue_b32(),input5, permNeigh12);
        input13 = svmax_s32_z(svptrue_b32(),input4, permNeigh13);
        input14 = svmax_s32_z(svptrue_b32(),input3, permNeigh14);
        input15 = svmax_s32_z(svptrue_b32(),input2, permNeigh15);
        input16 = svmax_s32_z(svptrue_b32(),input, permNeigh16);

        input8 = svmin_s32_z(svptrue_b32(),input8, permNeigh9);
        input7 = svmin_s32_z(svptrue_b32(),input7, permNeigh10);
        input6 = svmin_s32_z(svptrue_b32(),input6, permNeigh11);
        input5 = svmin_s32_z(svptrue_b32(),input5, permNeigh12);
        input4 = svmin_s32_z(svptrue_b32(),input4, permNeigh13);
        input3 = svmin_s32_z(svptrue_b32(),input3, permNeigh14);
        input2 = svmin_s32_z(svptrue_b32(),input2, permNeigh15);
        input = svmin_s32_z(svptrue_b32(),input, permNeigh16);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd8(input9, input10, input11, input12, input13, input14, input15, input16);
}

inline void CoreSmallSort16(int* __restrict__ ptr1, int* __restrict__ ptr2, int* __restrict__ ptr3, int* __restrict__ ptr4,
                             int* __restrict__ ptr5, int* __restrict__ ptr6, int* __restrict__ ptr7, int* __restrict__ ptr8,
                             int* __restrict__ ptr9, int* __restrict__ ptr10, int* __restrict__ ptr11, int* __restrict__ ptr12,
                             int* __restrict__ ptr13, int* __restrict__ ptr14, int* __restrict__ ptr15, int* __restrict__ ptr16){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr8);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr9);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr10);
    svint32_t input11 = svld1_s32(svptrue_b32(),ptr11);
    svint32_t input12 = svld1_s32(svptrue_b32(),ptr12);
    svint32_t input13 = svld1_s32(svptrue_b32(),ptr13);
    svint32_t input14 = svld1_s32(svptrue_b32(),ptr14);
    svint32_t input15 = svld1_s32(svptrue_b32(),ptr15);
    svint32_t input16 = svld1_s32(svptrue_b32(),ptr16);
    CoreSmallSort16(input1, input2, input3, input4, input5, input6, input7, input8,
                     input9, input10, input11, input12, input13, input14, input15, input16);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr2, input2);
    svst1_s32(svptrue_b32(),ptr3, input3);
    svst1_s32(svptrue_b32(),ptr4, input4);
    svst1_s32(svptrue_b32(),ptr5, input5);
    svst1_s32(svptrue_b32(),ptr6, input6);
    svst1_s32(svptrue_b32(),ptr7, input7);
    svst1_s32(svptrue_b32(),ptr8, input8);
    svst1_s32(svptrue_b32(),ptr9, input9);
    svst1_s32(svptrue_b32(),ptr10, input10);
    svst1_s32(svptrue_b32(),ptr11, input11);
    svst1_s32(svptrue_b32(),ptr12, input12);
    svst1_s32(svptrue_b32(),ptr13, input13);
    svst1_s32(svptrue_b32(),ptr14, input14);
    svst1_s32(svptrue_b32(),ptr15, input15);
    svst1_s32(svptrue_b32(),ptr16, input16);
}


////////////////////////////////////////////////////////////

inline void SmallSort16V(int* __restrict__ ptr, const size_t length){
    // length is limited to 4 times size of a vec
    const int nbValuesInVec = svcntw();
    const int nbVecs = (length+nbValuesInVec-1)/nbValuesInVec;
    const int rest = nbVecs*nbValuesInVec-length;
    const int lastVecSize = nbValuesInVec-rest;

    const svint32_t intMaxVector = svdup_s32(INT_MAX);
    const svbool_t maskRest = getTrueFalseMask32(lastVecSize);

    switch(nbVecs){
    case 1:
    {
        svint32_t v1 = svsel_s32(maskRest, svld1_s32(maskRest,ptr), intMaxVector);
        CoreSmallSort(v1);
        svst1_s32(maskRest, ptr, v1);
    }
        break;
    case 2:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v2 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec), intMaxVector);
        CoreSmallSort2(v1,v2);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(maskRest, ptr+nbValuesInVec, v2);
    }
        break;
    case 3:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v3 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*2), intMaxVector);
        CoreSmallSort3(v1,v2,v3);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(maskRest, ptr+nbValuesInVec*2, v3);
    }
        break;
    case 4:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v4 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*3), intMaxVector);
        CoreSmallSort4(v1,v2,v3,v4);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(maskRest, ptr+nbValuesInVec*3, v4);
    }
        break;
    case 5:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v5 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*4), intMaxVector);
        CoreSmallSort5(v1,v2,v3,v4,v5);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(maskRest, ptr+nbValuesInVec*4, v5);
    }
        break;
    case 6:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v6 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*5), intMaxVector);
        CoreSmallSort6(v1,v2,v3,v4,v5, v6);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(maskRest, ptr+nbValuesInVec*5, v6);
    }
        break;
    case 7:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v7 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*6), intMaxVector);
        CoreSmallSort7(v1,v2,v3,v4,v5,v6,v7);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(maskRest, ptr+nbValuesInVec*6, v7);
    }
        break;
    case 8:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v8 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*7), intMaxVector);
        CoreSmallSort8(v1,v2,v3,v4,v5,v6,v7,v8);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(maskRest, ptr+nbValuesInVec*7, v8);
    }
        break;
    case 9:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v8 = svld1_s32(svptrue_b32(),ptr+7*nbValuesInVec);
        svint32_t v9 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*8), intMaxVector);
        CoreSmallSort9(v1,v2,v3,v4,v5,v6,v7,v8,v9);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*7, v8);
        svst1_s32(maskRest, ptr+nbValuesInVec*8, v9);
    }
        break;
    case 10:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v8 = svld1_s32(svptrue_b32(),ptr+7*nbValuesInVec);
        svint32_t v9 = svld1_s32(svptrue_b32(),ptr+8*nbValuesInVec);
        svint32_t v10 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*9), intMaxVector);
        CoreSmallSort10(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*7, v8);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*8, v9);
        svst1_s32(maskRest, ptr+nbValuesInVec*9, v10);
    }
        break;
    case 11:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v8 = svld1_s32(svptrue_b32(),ptr+7*nbValuesInVec);
        svint32_t v9 = svld1_s32(svptrue_b32(),ptr+8*nbValuesInVec);
        svint32_t v10 = svld1_s32(svptrue_b32(),ptr+9*nbValuesInVec);
        svint32_t v11 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*10), intMaxVector);
        CoreSmallSort11(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*7, v8);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*8, v9);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*9, v10);
        svst1_s32(maskRest, ptr+nbValuesInVec*10, v11);
    }
        break;
    case 12:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v8 = svld1_s32(svptrue_b32(),ptr+7*nbValuesInVec);
        svint32_t v9 = svld1_s32(svptrue_b32(),ptr+8*nbValuesInVec);
        svint32_t v10 = svld1_s32(svptrue_b32(),ptr+9*nbValuesInVec);
        svint32_t v11 = svld1_s32(svptrue_b32(),ptr+10*nbValuesInVec);
        svint32_t v12 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*11), intMaxVector);
        CoreSmallSort12(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*7, v8);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*8, v9);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*9, v10);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*10, v11);
        svst1_s32(maskRest, ptr+nbValuesInVec*11, v12);
    }
        break;
    case 13:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v8 = svld1_s32(svptrue_b32(),ptr+7*nbValuesInVec);
        svint32_t v9 = svld1_s32(svptrue_b32(),ptr+8*nbValuesInVec);
        svint32_t v10 = svld1_s32(svptrue_b32(),ptr+9*nbValuesInVec);
        svint32_t v11 = svld1_s32(svptrue_b32(),ptr+10*nbValuesInVec);
        svint32_t v12 = svld1_s32(svptrue_b32(),ptr+11*nbValuesInVec);
        svint32_t v13 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*12), intMaxVector);
        CoreSmallSort13(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*7, v8);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*8, v9);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*9, v10);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*10, v11);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*11, v12);
        svst1_s32(maskRest, ptr+nbValuesInVec*12, v13);
    }
        break;
    case 14:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v8 = svld1_s32(svptrue_b32(),ptr+7*nbValuesInVec);
        svint32_t v9 = svld1_s32(svptrue_b32(),ptr+8*nbValuesInVec);
        svint32_t v10 = svld1_s32(svptrue_b32(),ptr+9*nbValuesInVec);
        svint32_t v11 = svld1_s32(svptrue_b32(),ptr+10*nbValuesInVec);
        svint32_t v12 = svld1_s32(svptrue_b32(),ptr+11*nbValuesInVec);
        svint32_t v13 = svld1_s32(svptrue_b32(),ptr+12*nbValuesInVec);
        svint32_t v14 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*13), intMaxVector);
        CoreSmallSort14(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*7, v8);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*8, v9);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*9, v10);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*10, v11);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*11, v12);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*12, v13);
        svst1_s32(maskRest, ptr+nbValuesInVec*13, v14);
    }
        break;
    case 15:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v8 = svld1_s32(svptrue_b32(),ptr+7*nbValuesInVec);
        svint32_t v9 = svld1_s32(svptrue_b32(),ptr+8*nbValuesInVec);
        svint32_t v10 = svld1_s32(svptrue_b32(),ptr+9*nbValuesInVec);
        svint32_t v11 = svld1_s32(svptrue_b32(),ptr+10*nbValuesInVec);
        svint32_t v12 = svld1_s32(svptrue_b32(),ptr+11*nbValuesInVec);
        svint32_t v13 = svld1_s32(svptrue_b32(),ptr+12*nbValuesInVec);
        svint32_t v14 = svld1_s32(svptrue_b32(),ptr+13*nbValuesInVec);
        svint32_t v15 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*14), intMaxVector);
        CoreSmallSort15(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*7, v8);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*8, v9);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*9, v10);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*10, v11);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*11, v12);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*12, v13);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*13, v14);
        svst1_s32(maskRest, ptr+nbValuesInVec*14, v15);
    }
        break;
        //case 16:
    default:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v8 = svld1_s32(svptrue_b32(),ptr+7*nbValuesInVec);
        svint32_t v9 = svld1_s32(svptrue_b32(),ptr+8*nbValuesInVec);
        svint32_t v10 = svld1_s32(svptrue_b32(),ptr+9*nbValuesInVec);
        svint32_t v11 = svld1_s32(svptrue_b32(),ptr+10*nbValuesInVec);
        svint32_t v12 = svld1_s32(svptrue_b32(),ptr+11*nbValuesInVec);
        svint32_t v13 = svld1_s32(svptrue_b32(),ptr+12*nbValuesInVec);
        svint32_t v14 = svld1_s32(svptrue_b32(),ptr+13*nbValuesInVec);
        svint32_t v15 = svld1_s32(svptrue_b32(),ptr+14*nbValuesInVec);
        svint32_t v16 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*15), intMaxVector);
        CoreSmallSort16(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*7, v8);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*8, v9);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*9, v10);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*10, v11);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*11, v12);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*12, v13);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*13, v14);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*14, v15);
        svst1_s32(maskRest, ptr+nbValuesInVec*15, v16);
    }
    }
}

inline void SmallSort16V(double* __restrict__ ptr, const size_t length){
    // length is limited to 4 times size of a vec
    const int nbValuesInVec = svcntd();
    const int nbVecs = (length+nbValuesInVec-1)/nbValuesInVec;
    const int rest = nbVecs*nbValuesInVec-length;
    const int lastVecSize = nbValuesInVec-rest;

    const svfloat64_t doubleMaxVector = svdup_f64(DBL_MAX);
    const svbool_t maskRest = getTrueFalseMask64(lastVecSize);

    switch(nbVecs){
    case 1:
    {
        svfloat64_t v1 = svsel_f64(maskRest, svld1_f64(maskRest,ptr), doubleMaxVector);
        CoreSmallSort(v1);
        svst1_f64(maskRest, ptr, v1);
    }
        break;
    case 2:
    {
        svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr);
        svfloat64_t v2 = svsel_f64(maskRest, svld1_f64(maskRest,ptr+nbValuesInVec), doubleMaxVector);
        CoreSmallSort2(v1,v2);
        svst1_f64(svptrue_b64(), ptr, v1);
        svst1_f64(maskRest, ptr+nbValuesInVec, v2);
    }
        break;
    case 3:
    {
        svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr);
        svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr+nbValuesInVec);
        svfloat64_t v3 = svsel_f64(maskRest, svld1_f64(maskRest,ptr+nbValuesInVec*2), doubleMaxVector);
        CoreSmallSort3(v1,v2,v3);
        svst1_f64(svptrue_b64(), ptr, v1);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec, v2);
        svst1_f64(maskRest, ptr+nbValuesInVec*2, v3);
    }
        break;
    case 4:
    {
        svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr);
        svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr+nbValuesInVec);
        svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr+2*nbValuesInVec);
        svfloat64_t v4 = svsel_f64(maskRest, svld1_f64(maskRest,ptr+nbValuesInVec*3), doubleMaxVector);
        CoreSmallSort4(v1,v2,v3,v4);
        svst1_f64(svptrue_b64(), ptr, v1);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec, v2);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*2, v3);
        svst1_f64(maskRest, ptr+nbValuesInVec*3, v4);
    }
        break;
    case 5:
    {
        svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr);
        svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr+nbValuesInVec);
        svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr+2*nbValuesInVec);
        svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr+3*nbValuesInVec);
        svfloat64_t v5 = svsel_f64(maskRest, svld1_f64(maskRest,ptr+nbValuesInVec*4), doubleMaxVector);
        CoreSmallSort5(v1,v2,v3,v4,v5);
        svst1_f64(svptrue_b64(), ptr, v1);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec, v2);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*2, v3);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*3, v4);
        svst1_f64(maskRest, ptr+nbValuesInVec*4, v5);
    }
        break;
    case 6:
    {
        svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr);
        svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr+nbValuesInVec);
        svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr+2*nbValuesInVec);
        svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr+3*nbValuesInVec);
        svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr+4*nbValuesInVec);
        svfloat64_t v6 = svsel_f64(maskRest, svld1_f64(maskRest,ptr+nbValuesInVec*5), doubleMaxVector);
        CoreSmallSort6(v1,v2,v3,v4,v5, v6);
        svst1_f64(svptrue_b64(), ptr, v1);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec, v2);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*2, v3);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*3, v4);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*4, v5);
        svst1_f64(maskRest, ptr+nbValuesInVec*5, v6);
    }
        break;
    case 7:
    {
        svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr);
        svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr+nbValuesInVec);
        svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr+2*nbValuesInVec);
        svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr+3*nbValuesInVec);
        svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr+4*nbValuesInVec);
        svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr+5*nbValuesInVec);
        svfloat64_t v7 = svsel_f64(maskRest, svld1_f64(maskRest,ptr+nbValuesInVec*6), doubleMaxVector);
        CoreSmallSort7(v1,v2,v3,v4,v5,v6,v7);
        svst1_f64(svptrue_b64(), ptr, v1);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec, v2);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*2, v3);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*3, v4);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*4, v5);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*5, v6);
        svst1_f64(maskRest, ptr+nbValuesInVec*6, v7);
    }
        break;
    case 8:
    {
        svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr);
        svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr+nbValuesInVec);
        svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr+2*nbValuesInVec);
        svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr+3*nbValuesInVec);
        svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr+4*nbValuesInVec);
        svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr+5*nbValuesInVec);
        svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr+6*nbValuesInVec);
        svfloat64_t v8 = svsel_f64(maskRest, svld1_f64(maskRest,ptr+nbValuesInVec*7), doubleMaxVector);
        CoreSmallSort8(v1,v2,v3,v4,v5,v6,v7,v8);
        svst1_f64(svptrue_b64(), ptr, v1);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec, v2);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*2, v3);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*3, v4);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*4, v5);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*5, v6);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*6, v7);
        svst1_f64(maskRest, ptr+nbValuesInVec*7, v8);
    }
        break;
    case 9:
    {
        svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr);
        svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr+nbValuesInVec);
        svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr+2*nbValuesInVec);
        svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr+3*nbValuesInVec);
        svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr+4*nbValuesInVec);
        svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr+5*nbValuesInVec);
        svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr+6*nbValuesInVec);
        svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr+7*nbValuesInVec);
        svfloat64_t v9 = svsel_f64(maskRest, svld1_f64(maskRest,ptr+nbValuesInVec*8), doubleMaxVector);
        CoreSmallSort9(v1,v2,v3,v4,v5,v6,v7,v8,v9);
        svst1_f64(svptrue_b64(), ptr, v1);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec, v2);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*2, v3);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*3, v4);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*4, v5);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*5, v6);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*6, v7);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*7, v8);
        svst1_f64(maskRest, ptr+nbValuesInVec*8, v9);
    }
        break;
    case 10:
    {
        svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr);
        svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr+nbValuesInVec);
        svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr+2*nbValuesInVec);
        svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr+3*nbValuesInVec);
        svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr+4*nbValuesInVec);
        svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr+5*nbValuesInVec);
        svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr+6*nbValuesInVec);
        svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr+7*nbValuesInVec);
        svfloat64_t v9 = svld1_f64(svptrue_b64(),ptr+8*nbValuesInVec);
        svfloat64_t v10 = svsel_f64(maskRest, svld1_f64(maskRest,ptr+nbValuesInVec*9), doubleMaxVector);
        CoreSmallSort10(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10);
        svst1_f64(svptrue_b64(), ptr, v1);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec, v2);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*2, v3);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*3, v4);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*4, v5);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*5, v6);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*6, v7);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*7, v8);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*8, v9);
        svst1_f64(maskRest, ptr+nbValuesInVec*9, v10);
    }
        break;
    case 11:
    {
        svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr);
        svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr+nbValuesInVec);
        svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr+2*nbValuesInVec);
        svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr+3*nbValuesInVec);
        svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr+4*nbValuesInVec);
        svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr+5*nbValuesInVec);
        svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr+6*nbValuesInVec);
        svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr+7*nbValuesInVec);
        svfloat64_t v9 = svld1_f64(svptrue_b64(),ptr+8*nbValuesInVec);
        svfloat64_t v10 = svld1_f64(svptrue_b64(),ptr+9*nbValuesInVec);
        svfloat64_t v11 = svsel_f64(maskRest, svld1_f64(maskRest,ptr+nbValuesInVec*10), doubleMaxVector);
        CoreSmallSort11(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11);
        svst1_f64(svptrue_b64(), ptr, v1);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec, v2);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*2, v3);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*3, v4);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*4, v5);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*5, v6);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*6, v7);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*7, v8);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*8, v9);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*9, v10);
        svst1_f64(maskRest, ptr+nbValuesInVec*10, v11);
    }
        break;
    case 12:
    {
        svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr);
        svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr+nbValuesInVec);
        svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr+2*nbValuesInVec);
        svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr+3*nbValuesInVec);
        svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr+4*nbValuesInVec);
        svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr+5*nbValuesInVec);
        svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr+6*nbValuesInVec);
        svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr+7*nbValuesInVec);
        svfloat64_t v9 = svld1_f64(svptrue_b64(),ptr+8*nbValuesInVec);
        svfloat64_t v10 = svld1_f64(svptrue_b64(),ptr+9*nbValuesInVec);
        svfloat64_t v11 = svld1_f64(svptrue_b64(),ptr+10*nbValuesInVec);
        svfloat64_t v12 = svsel_f64(maskRest, svld1_f64(maskRest,ptr+nbValuesInVec*11), doubleMaxVector);
        CoreSmallSort12(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12);
        svst1_f64(svptrue_b64(), ptr, v1);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec, v2);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*2, v3);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*3, v4);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*4, v5);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*5, v6);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*6, v7);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*7, v8);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*8, v9);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*9, v10);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*10, v11);
        svst1_f64(maskRest, ptr+nbValuesInVec*11, v12);
    }
        break;
    case 13:
    {
        svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr);
        svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr+nbValuesInVec);
        svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr+2*nbValuesInVec);
        svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr+3*nbValuesInVec);
        svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr+4*nbValuesInVec);
        svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr+5*nbValuesInVec);
        svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr+6*nbValuesInVec);
        svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr+7*nbValuesInVec);
        svfloat64_t v9 = svld1_f64(svptrue_b64(),ptr+8*nbValuesInVec);
        svfloat64_t v10 = svld1_f64(svptrue_b64(),ptr+9*nbValuesInVec);
        svfloat64_t v11 = svld1_f64(svptrue_b64(),ptr+10*nbValuesInVec);
        svfloat64_t v12 = svld1_f64(svptrue_b64(),ptr+11*nbValuesInVec);
        svfloat64_t v13 = svsel_f64(maskRest, svld1_f64(maskRest,ptr+nbValuesInVec*12), doubleMaxVector);
        CoreSmallSort13(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13);
        svst1_f64(svptrue_b64(), ptr, v1);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec, v2);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*2, v3);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*3, v4);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*4, v5);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*5, v6);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*6, v7);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*7, v8);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*8, v9);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*9, v10);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*10, v11);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*11, v12);
        svst1_f64(maskRest, ptr+nbValuesInVec*12, v13);
    }
        break;
    case 14:
    {
        svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr);
        svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr+nbValuesInVec);
        svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr+2*nbValuesInVec);
        svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr+3*nbValuesInVec);
        svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr+4*nbValuesInVec);
        svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr+5*nbValuesInVec);
        svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr+6*nbValuesInVec);
        svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr+7*nbValuesInVec);
        svfloat64_t v9 = svld1_f64(svptrue_b64(),ptr+8*nbValuesInVec);
        svfloat64_t v10 = svld1_f64(svptrue_b64(),ptr+9*nbValuesInVec);
        svfloat64_t v11 = svld1_f64(svptrue_b64(),ptr+10*nbValuesInVec);
        svfloat64_t v12 = svld1_f64(svptrue_b64(),ptr+11*nbValuesInVec);
        svfloat64_t v13 = svld1_f64(svptrue_b64(),ptr+12*nbValuesInVec);
        svfloat64_t v14 = svsel_f64(maskRest, svld1_f64(maskRest,ptr+nbValuesInVec*13), doubleMaxVector);
        CoreSmallSort14(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14);
        svst1_f64(svptrue_b64(), ptr, v1);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec, v2);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*2, v3);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*3, v4);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*4, v5);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*5, v6);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*6, v7);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*7, v8);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*8, v9);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*9, v10);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*10, v11);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*11, v12);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*12, v13);
        svst1_f64(maskRest, ptr+nbValuesInVec*13, v14);
    }
        break;
    case 15:
    {
        svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr);
        svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr+nbValuesInVec);
        svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr+2*nbValuesInVec);
        svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr+3*nbValuesInVec);
        svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr+4*nbValuesInVec);
        svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr+5*nbValuesInVec);
        svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr+6*nbValuesInVec);
        svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr+7*nbValuesInVec);
        svfloat64_t v9 = svld1_f64(svptrue_b64(),ptr+8*nbValuesInVec);
        svfloat64_t v10 = svld1_f64(svptrue_b64(),ptr+9*nbValuesInVec);
        svfloat64_t v11 = svld1_f64(svptrue_b64(),ptr+10*nbValuesInVec);
        svfloat64_t v12 = svld1_f64(svptrue_b64(),ptr+11*nbValuesInVec);
        svfloat64_t v13 = svld1_f64(svptrue_b64(),ptr+12*nbValuesInVec);
        svfloat64_t v14 = svld1_f64(svptrue_b64(),ptr+13*nbValuesInVec);
        svfloat64_t v15 = svsel_f64(maskRest, svld1_f64(maskRest,ptr+nbValuesInVec*14), doubleMaxVector);
        CoreSmallSort15(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15);
        svst1_f64(svptrue_b64(), ptr, v1);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec, v2);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*2, v3);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*3, v4);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*4, v5);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*5, v6);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*6, v7);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*7, v8);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*8, v9);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*9, v10);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*10, v11);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*11, v12);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*12, v13);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*13, v14);
        svst1_f64(maskRest, ptr+nbValuesInVec*14, v15);
    }
        break;
        //case 16:
    default:
    {
        svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr);
        svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr+nbValuesInVec);
        svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr+2*nbValuesInVec);
        svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr+3*nbValuesInVec);
        svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr+4*nbValuesInVec);
        svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr+5*nbValuesInVec);
        svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr+6*nbValuesInVec);
        svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr+7*nbValuesInVec);
        svfloat64_t v9 = svld1_f64(svptrue_b64(),ptr+8*nbValuesInVec);
        svfloat64_t v10 = svld1_f64(svptrue_b64(),ptr+9*nbValuesInVec);
        svfloat64_t v11 = svld1_f64(svptrue_b64(),ptr+10*nbValuesInVec);
        svfloat64_t v12 = svld1_f64(svptrue_b64(),ptr+11*nbValuesInVec);
        svfloat64_t v13 = svld1_f64(svptrue_b64(),ptr+12*nbValuesInVec);
        svfloat64_t v14 = svld1_f64(svptrue_b64(),ptr+13*nbValuesInVec);
        svfloat64_t v15 = svld1_f64(svptrue_b64(),ptr+14*nbValuesInVec);
        svfloat64_t v16 = svsel_f64(maskRest, svld1_f64(maskRest,ptr+nbValuesInVec*15), doubleMaxVector);
        CoreSmallSort16(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16);
        svst1_f64(svptrue_b64(), ptr, v1);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec, v2);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*2, v3);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*3, v4);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*4, v5);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*5, v6);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*6, v7);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*7, v8);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*8, v9);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*9, v10);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*10, v11);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*11, v12);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*12, v13);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*13, v14);
        svst1_f64(svptrue_b64(), ptr+nbValuesInVec*14, v15);
        svst1_f64(maskRest, ptr+nbValuesInVec*15, v16);
    }
    }
}


template <class SortType, class IndexType>
static inline IndexType CoreScalarPartition(SortType array[], IndexType left, IndexType right,
                                            const SortType pivot){

    for(; left <= right
        && array[left] <= pivot ; ++left){
    }

    for(IndexType idx = left ; idx <= right ; ++idx){
        if( array[idx] <= pivot ){
            std::swap(array[idx],array[left]);
            left += 1;
        }
    }

    return left;
}

/* a sequential qs */
template <class IndexType>
static inline IndexType PartitionSVE(int array[], IndexType left, IndexType right,
                                     const int pivot){
    const IndexType S = svcntw();

    if(right-left+1 < 2*S){
        return CoreScalarPartition<int,IndexType>(array, left, right, pivot);
    }

    svint32_t pivotvec = svdup_s32(pivot);

    svint32_t left_val = svld1_s32(svptrue_b32(),&array[left]);
    IndexType left_w = left;
    left += S;

    IndexType right_w = right+1;
    right -= S-1;
    svint32_t right_val = svld1_s32(svptrue_b32(),&array[right]);

    while(left + S <= right){
        const IndexType free_left = left - left_w;
        const IndexType free_right = right_w - right;

        svint32_t val;
        if( free_left <= free_right ){
            val = svld1_s32(svptrue_b32(),&array[left]);
            left += S;
        }
        else{
            right -= S;
            val = svld1_s32(svptrue_b32(),&array[right]);
        }

        svbool_t mask = svcmple_s32(svptrue_b32(), val, pivotvec);

        const IndexType nb_low = svcntp_b32(svptrue_b32(),mask);
        const IndexType nb_high = S-nb_low;

        svint32_t val_low = svcompact_s32(mask, val);
        svint32_t val_high = svcompact_s32(svnot_b_z(svptrue_b32(), mask), val);

        svst1_s32(getTrueFalseMask32(nb_low),&array[left_w],val_low);
        left_w += nb_low;

        right_w -= nb_high;
        svst1_s32(getTrueFalseMask32(nb_high),&array[right_w],val_high);
    }

    {
        const IndexType remaining = right - left;
        svbool_t remainingMask = getTrueFalseMask32(remaining);
        svint32_t val = svld1_s32(remainingMask,&array[left]);
        left = right;

        svbool_t mask_low = svcmple_s32(remainingMask, val, pivotvec);
        svbool_t mask_high = svnot_b_z(remainingMask, mask_low);

        const IndexType nb_low = svcntp_b32(remainingMask,mask_low);
        const IndexType nb_high = svcntp_b32(remainingMask,mask_high);

        svint32_t val_low = svcompact_s32(mask_low, val);
        svint32_t val_high = svcompact_s32(mask_high, val);

        svst1_s32(getTrueFalseMask32(nb_low),&array[left_w],val_low);
        left_w += nb_low;

        right_w -= nb_high;
        svst1_s32(getTrueFalseMask32(nb_high),&array[right_w],val_high);
    }
    {
        svbool_t mask = svcmple_s32(svptrue_b32(), left_val, pivotvec);

        const IndexType nb_low = svcntp_b32(svptrue_b32(),mask);
        const IndexType nb_high = S-nb_low;

        svint32_t val_low = svcompact_s32(mask, left_val);
        svint32_t val_high = svcompact_s32(svnot_b_z(svptrue_b32(), mask), left_val);

        svst1_s32(getTrueFalseMask32(nb_low),&array[left_w],val_low);
        left_w += nb_low;

        right_w -= nb_high;
        svst1_s32(getTrueFalseMask32(nb_high),&array[right_w],val_high);
    }
    {
        svbool_t mask = svcmple_s32(svptrue_b32(), right_val, pivotvec);

        const IndexType nb_low = svcntp_b32(svptrue_b32(),mask);
        const IndexType nb_high = S-nb_low;

        svint32_t val_low = svcompact_s32(mask, right_val);
        svint32_t val_high = svcompact_s32(svnot_b_z(svptrue_b32(), mask), right_val);

        svst1_s32(getTrueFalseMask32(nb_low),&array[left_w],val_low);
        left_w += nb_low;

        right_w -= nb_high;
        svst1_s32(getTrueFalseMask32(nb_high),&array[right_w],val_high);
    }
    return left_w;
}

template <class IndexType>
static inline IndexType PartitionSVE(double array[], IndexType left, IndexType right,
                                     const double pivot){
    const IndexType S = svcntd();

    if(right-left+1 < 2*S){
        return CoreScalarPartition<double,IndexType>(array, left, right, pivot);
    }

    svfloat64_t pivotvec = svdup_f64(pivot);

    svfloat64_t left_val = svld1_f64(svptrue_b64(),&array[left]);
    IndexType left_w = left;
    left += S;

    IndexType right_w = right+1;
    right -= S-1;
    svfloat64_t right_val = svld1_f64(svptrue_b64(),&array[right]);

    while(left + S <= right){
        const IndexType free_left = left - left_w;
        const IndexType free_right = right_w - right;

        svfloat64_t val;
        if( free_left <= free_right ){
            val = svld1_f64(svptrue_b64(),&array[left]);
            left += S;
        }
        else{
            right -= S;
            val = svld1_f64(svptrue_b64(),&array[right]);
        }

        svbool_t mask = svcmple_f64(svptrue_b64(), val, pivotvec);

        const IndexType nb_low = svcntp_b64(svptrue_b64(),mask);
        const IndexType nb_high = S-nb_low;

        svfloat64_t val_low = svcompact_f64(mask, val);
        svfloat64_t val_high = svcompact_f64(svnot_b_z(svptrue_b64(), mask), val);

        svst1_f64(getTrueFalseMask64(nb_low),&array[left_w],val_low);
        left_w += nb_low;

        right_w -= nb_high;
        svst1_f64(getTrueFalseMask64(nb_high),&array[right_w],val_high);
    }

    {
        const IndexType remaining = right - left;
        svbool_t remainingMask = getTrueFalseMask64(remaining);
                svfloat64_t val = svld1_f64(remainingMask,&array[left]);
        left = right;

        svbool_t mask_low = svcmple_f64(remainingMask, val, pivotvec);
        svbool_t mask_high = svnot_b_z(remainingMask, mask_low);

        const IndexType nb_low = svcntp_b64(remainingMask,mask_low);
        const IndexType nb_high = svcntp_b64(remainingMask,mask_high);

        svfloat64_t val_low = svcompact_f64(mask_low, val);
        svfloat64_t val_high = svcompact_f64(mask_high, val);

        svst1_f64(getTrueFalseMask64(nb_low),&array[left_w],val_low);
        left_w += nb_low;

        right_w -= nb_high;
        svst1_f64(getTrueFalseMask64(nb_high),&array[right_w],val_high);
    }
    {
        svbool_t mask = svcmple_f64(svptrue_b64(), left_val, pivotvec);

        const IndexType nb_low = svcntp_b64(svptrue_b64(),mask);
        const IndexType nb_high = S-nb_low;

        svfloat64_t val_low = svcompact_f64(mask, left_val);
        svfloat64_t val_high = svcompact_f64(svnot_b_z(svptrue_b64(), mask), left_val);

        svst1_f64(getTrueFalseMask64(nb_low),&array[left_w],val_low);
        left_w += nb_low;

        right_w -= nb_high;
        svst1_f64(getTrueFalseMask64(nb_high),&array[right_w],val_high);
    }
    {
        svbool_t mask = svcmple_f64(svptrue_b64(), right_val, pivotvec);

        const IndexType nb_low = svcntp_b64(svptrue_b64(),mask);
        const IndexType nb_high = S-nb_low;

        svfloat64_t val_low = svcompact_f64(mask, right_val);
        svfloat64_t val_high = svcompact_f64(svnot_b_z(svptrue_b64(), mask), right_val);

        svst1_f64(getTrueFalseMask64(nb_low),&array[left_w],val_low);
        left_w += nb_low;

        right_w -= nb_high;
        svst1_f64(getTrueFalseMask64(nb_high),&array[right_w],val_high);
    }
    return left_w;
}

template <class SortType, class IndexType = size_t>
static inline IndexType CoreSortGetPivot(const SortType array[], const IndexType left, const IndexType right){
    const IndexType middle = ((right-left)/2) + left;
    if(array[left] <= array[middle] && array[middle] <= array[right]){
        return middle;
    }
    else if(array[middle] <= array[left] && array[left] <= array[right]){
        return left;
    }
    else return right;
}


template <class SortType, class IndexType = size_t>
static inline IndexType CoreSortPivotPartition(SortType array[], const IndexType left, const IndexType right){
    if(right-left > 1){
        const IndexType pivotIdx = CoreSortGetPivot(array, left, right);
        std::swap(array[pivotIdx], array[right]);
        const IndexType part = PartitionSVE(array, left, right-1, array[right]);
        std::swap(array[part], array[right]);
        return part;
    }
    return left;
}

template <class SortType, class IndexType = size_t>
static inline IndexType CoreSortPartition(SortType array[], const IndexType left, const IndexType right,
                                          const SortType pivot){
    return  PartitionSVE(array, left, right, pivot);
}

template <class SortType, class IndexType = size_t>
static void CoreSort(SortType array[], const IndexType left, const IndexType right){
    static const IndexType SortLimite = 16*svcntb()/sizeof(SortType);
    if(right-left < SortLimite){
        SmallSort16V(array+left, right-left+1);
    }
    else{
        const IndexType part = CoreSortPivotPartition<SortType,IndexType>(array, left, right);
        if(part+1 < right) CoreSort<SortType,IndexType>(array,part+1,right);
        auto what = part - 1;
        if(part && left < part-1)  
            CoreSort<SortType,IndexType>(array,left,what);
            // CoreSort<SortType,IndexType>(array,left,part - 1);
    }
}

template <class SortType, class IndexType = size_t>
static inline void Sort(SortType array[], const IndexType size){
    static const IndexType SortLimite = 16*svcntb()/sizeof(SortType);
    if(size <= SortLimite){
        SmallSort16V(array, size);
        return;
    }
    CoreSort<SortType,IndexType>(array, 0, size-1);
}


#if defined(_OPENMP)

template <class IndexType = size_t>
struct Interval{
    IndexType left;
    IndexType right;
};

template <class IndexType = size_t>
struct Bucket{
    std::deque<Interval<IndexType>> tasks;
    omp_lock_t taskLock;
    int taskCounter;
    
    // https://www.fujitsu.com/downloads/SUPER/a64fx/a64fx_datasheet.pdf
    static const int CacheLineSize = 256;
    static const int SizeOfAttributs = sizeof(std::deque<Interval<IndexType>>)+sizeof(omp_lock_t)+sizeof(int);
    static const int Space = (CacheLineSize - SizeOfAttributs%CacheLineSize);
    unsigned char padding[Space];
};



template <class SortType, class IndexType = size_t>
static inline void CoreSortTaskPartition(SortType array[], IndexType left, IndexType right, Bucket<IndexType>* buckets){
    static const IndexType SortLimite = 16*svcntb()/sizeof(SortType);
    static const IndexType L1Limite = 64*1024/sizeof(SortType);
    const int idxThread = omp_get_thread_num();
    int& refCounter = buckets[idxThread].taskCounter;
    
    while(true){
        if(right-left < SortLimite){
            SmallSort16V(array+left, right-left+1);
            break;
        }
        else{
            const IndexType part = CoreSortPivotPartition<SortType,IndexType>(array, left, right);

            if( part+1 < right && part && left < part-1
                 && ((right-part-1) > L1Limite || (part-1-left) > L1Limite)){
                if(right-part > part-left){                    
                    omp_set_lock(&buckets[idxThread].taskLock);
                    buckets[idxThread].tasks.push_front(Interval<IndexType>{left, part - 1});
                    #pragma omp atomic update
                    refCounter += 1;
                    omp_unset_lock(&buckets[idxThread].taskLock);
                    left = part+1;
                }                                        
                else{                  
                    omp_set_lock(&buckets[idxThread].taskLock);
                    buckets[idxThread].tasks.push_front(Interval<IndexType>{part+1,right});
                    #pragma omp atomic update
                    refCounter += 1;
                    omp_unset_lock(&buckets[idxThread].taskLock);
                    right = part-1;
                }
            }
            else {
                if(part+1 < right) CoreSort<SortType,IndexType>(array,part+1,right);
                if(part && left < part-1)  CoreSort<SortType,IndexType>(array,left,part - 1);
                break;
            }
        }
    }
}

template <class SortType, class IndexType = size_t>
static inline void SortOmpPartition(SortType array[], const IndexType size){
    const IndexType L1Limite = 64*1024/sizeof(SortType);
    if(omp_get_max_threads() == 1 || size < L1Limite){
        Sort(array, size);
        return;
    }

    const int MaxThreads = 128;
    Bucket<IndexType> buckets[MaxThreads];

    const int NbThreadsToUse = std::min(omp_get_max_threads(), int((size+L1Limite)/L1Limite));
    for(int idxThread = 0 ; idxThread < NbThreadsToUse ; ++idxThread){
        omp_init_lock(&buckets[idxThread].taskLock);
        buckets[idxThread].taskCounter = 0;
    }

    int counterIdle = 0;

#pragma omp parallel default(shared) num_threads(NbThreadsToUse)
    {
#pragma omp master
        {
            CoreSortTaskPartition<SortType,IndexType>(array, 0, size-1, buckets);
        }

        bool isIdle = false;

        while(true){
            bool hasOne = false;
            Interval<IndexType> currentInterval;

            for(int idxThreadOther = 0 ; idxThreadOther < omp_get_num_threads() && hasOne == false ; ++idxThreadOther){
                const int idxThread = (idxThreadOther+(idxThreadOther%2?((idxThreadOther+1)/2):(-(idxThreadOther+1)/2))+omp_get_num_threads())%omp_get_num_threads();
                int& refCounter = buckets[idxThread].taskCounter;
                int currentValue;
                #pragma omp atomic read
                currentValue = refCounter;
                if(currentValue){
                    omp_set_lock(&buckets[idxThread].taskLock);
                    if(buckets[idxThread].tasks.size()){
                        if(idxThread == omp_get_thread_num()){
                            currentInterval = buckets[idxThread].tasks.front();
                            buckets[idxThread].tasks.pop_front();
                        }
                        else{
                            currentInterval = buckets[idxThread].tasks.back();
                            buckets[idxThread].tasks.pop_back();
                        }
                        hasOne = true;
                        #pragma omp atomic update
                        refCounter -= 1;
                        if(isIdle){
                            isIdle = false;
                            #pragma omp atomic update
                            counterIdle -= 1;
                        }
                    }
                    omp_unset_lock(&buckets[idxThread].taskLock);
                }

            }

            if(hasOne){
                CoreSortTaskPartition<SortType,IndexType>(array, currentInterval.left, currentInterval.right, buckets);
            }
            else{
                if(isIdle == false){
                    isIdle = true;
                    #pragma omp atomic update
                    counterIdle += 1;
                }

                int currentIldeValue;
                #pragma omp atomic read
                currentIldeValue = counterIdle;
                if(currentIldeValue == omp_get_num_threads()){
                    break;
                }
            }
        }
    }

    for(int idxThread = 0 ; idxThread < NbThreadsToUse ; ++idxThread){
        omp_destroy_lock(&buckets[idxThread].taskLock);
    }
}

#endif
}


#endif
