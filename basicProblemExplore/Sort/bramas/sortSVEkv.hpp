//////////////////////////////////////////////////////////
/// By berenger.bramas@inria.fr 2020.
/// Licence is MIT.
/// Comes without any warranty.
///
/// Code to sort two arrays of integers (key/value)
/// or sort pairs of integers
/// using ARM SVE (works for vectors of any size).
/// It also includes a partitioning function.
///
/// Please refer to the README to know how to build
/// and to have more information about the functions.
///
//////////////////////////////////////////////////////////
#ifndef SORTSVEKV_HPP
#define SORTSVEKV_HPP

#ifndef __ARM_FEATURE_SVE
#warning __ARM_FEATURE_SVE undefined
#endif
#include <arm_sve.h>

#include <climits>
#include <cfloat>
#include <algorithm>

#if defined(_OPENMP)
#include <omp.h>
#else
#warning OpenMP disabled
#endif

static_assert (sizeof(std::pair<int,int>) == sizeof(int)*2, "Must be true");

namespace SortSVEkv {


inline void CoreExchange(svint32_t& input, svint32_t& input2,
                         svint32_t& input_val, svint32_t& input2_val){
    svint32_t permNeigh2 = svrev_s32(input2);
    svint32_t permNeigh2_val = svrev_s32(input2_val);
    svbool_t mask = svcmplt_s32(svptrue_b32(), input, permNeigh2);
    input2 = svsel_s32(mask, permNeigh2, input);// max
    input = svsel_s32(mask, input, permNeigh2);// min
    input2_val = svsel_s32(mask, permNeigh2_val, input_val);// max
    input_val = svsel_s32(mask, input_val, permNeigh2_val);// min
}

inline void CoreExchangeEnd(svint32_t& input, svint32_t& input2,
                            svint32_t& input_val, svint32_t& input2_val){
    svint32_t inputCopy = input;
    svint32_t inputCopy_val = input_val;
    svbool_t mask = svcmplt_s32(svptrue_b32(), input, input2);
    input = svsel_s32(mask, input, input2);// min
    input2 = svsel_s32(mask, input2, inputCopy);// max
    input_val = svsel_s32(mask, input_val, input2_val);// min
    input2_val = svsel_s32(mask, input2_val, inputCopy_val);// max
}


inline void CoreSmallSort(svint32_t& input,
                          svint32_t& input_val){
    const int nbValuesInVec = svcntw();

    const svint32_t vecindex = svindex_s32(0, 1);

    svbool_t ffttout = svzip1_b32(svpfalse_b(),svptrue_b32());

    svint32_t vecincout = svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

    {// stepout == 1
        const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

        const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
        const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

        svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

        input = svsel_s32(ffttout,
                          svsel_s32(mask, inputPerm, input),// max
                          svsel_s32(mask, input, inputPerm));// min

        input_val = svsel_s32(ffttout,
                          svsel_s32(mask, inputPerm_val, input_val),// max
                          svsel_s32(mask, input_val, inputPerm_val));// min
    }
    for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
        ffttout = svzip1_b32(ffttout,ffttout);
        vecincout = svsel_s32(ffttout,
                                  svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                                  svadd_n_s32_z(svptrue_b32(), vecincout, stepout));

        {
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

            input = svsel_s32(ffttout,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(ffttout,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min
        }

        svbool_t fftt = svuzp2_b32(ffttout,ffttout);

        svint32_t vecinc = svdup_s32(stepout/2);

        for(long int step = stepout/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min
        }
    }
}


inline void CoreSmallSortEnd(svint32_t& input,
                             svint32_t& input_val){
    const int nbValuesInVec = svcntw();

    svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

    const svint32_t vecindex = svindex_s32(0, 1);

    svint32_t vecinc = svdup_s32(nbValuesInVec/2);

    for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
        const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

        const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
        const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

        svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

        input = svsel_s32(fftt,
                          svsel_s32(mask, inputPerm, input),// max
                          svsel_s32(mask, input, inputPerm));// min

        input_val = svsel_s32(fftt,
                          svsel_s32(mask, inputPerm_val, input_val),// max
                          svsel_s32(mask, input_val, inputPerm_val));// min

        fftt = svuzp2_b32(fftt,fftt);
        vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
    }
    { // Step == 1
        const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

        const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
        const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

        svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

        input = svsel_s32(fftt,
                          svsel_s32(mask, inputPerm, input),// max
                          svsel_s32(mask, input, inputPerm));// min

        input_val = svsel_s32(fftt,
                          svsel_s32(mask, inputPerm_val, input_val),// max
                          svsel_s32(mask, input_val, inputPerm_val));// min
    }
}



inline void CoreSmallEnd2(svint32_t& input, svint32_t& input2,
                          svint32_t& input_val, svint32_t& input2_val){
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
    }
#ifdef NOOPTIM
    CoreSmallSortEnd(input,
                  input_val);
    CoreSmallSortEnd(input2,
                  input2_val);
#else
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
        }
    }
#endif
}

inline void CoreSmallSort2(svint32_t& input, svint32_t& input2,
                           svint32_t& input_val, svint32_t& input2_val){
#ifdef NOOPTIM
    CoreSmallSort(input,
                  input_val);
    CoreSmallSort(input2,
                  input2_val);
#else
    {
        const int nbValuesInVec = svcntw();

        const svint32_t vecindex = svindex_s32(0, 1);

        svbool_t ffttout = svzip1_b32(svpfalse_b(),svptrue_b32());

        svint32_t vecincout = svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

        {// stepout == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

            input = svsel_s32(ffttout,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(ffttout,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min

            const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

            input2 = svsel_s32(ffttout,
                              svsel_s32(mask2, input2Perm, input2),// max
                              svsel_s32(mask2, input2, input2Perm));// min

            input2_val = svsel_s32(ffttout,
                              svsel_s32(mask2, input2Perm_val, input2_val),// max
                              svsel_s32(mask2, input2_val, input2Perm_val));// min
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b32(ffttout,ffttout);
            vecincout = svsel_s32(ffttout,
                                      svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                                      svadd_n_s32_z(svptrue_b32(), vecincout, stepout));

            {
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(ffttout,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(ffttout,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(ffttout,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(ffttout,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min
            }

            svbool_t fftt = svuzp2_b32(ffttout,ffttout);

            svint32_t vecinc = svdup_s32(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                fftt = svuzp2_b32(fftt,fftt);
                vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min
            }
        }
    }
#endif
    {
        CoreExchange(input, input2,
                     input_val, input2_val);
    }
#ifdef NOOPTIM
    CoreSmallSortEnd(input,
                 input_val);
    CoreSmallSortEnd(input2,
                 input2_val);
#else
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
        }
    }
#endif
}

inline void CoreSmallEnd3(svint32_t& input, svint32_t& input2, svint32_t& input3,
                          svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val){
    {
        CoreExchangeEnd(input, input3,
                        input_val, input3_val);
    }
#ifdef NOOPTIM
    CoreSmallEnd2(input, input2,
                   input_val, input2_val);
    CoreSmallSortEnd(input3,
                  input3_val);
#else
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
        }
    }
#endif
}

inline void CoreSmallSort3(svint32_t& input, svint32_t& input2, svint32_t& input3,
                           svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val){
#ifdef NOOPTIM
    CoreSmallSort2(input, input2,
                   input_val, input2_val);
    CoreSmallSort(input3,
                  input3_val);
#else
    {
        const int nbValuesInVec = svcntw();

        const svint32_t vecindex = svindex_s32(0, 1);

        svbool_t ffttout = svzip1_b32(svpfalse_b(),svptrue_b32());

        svint32_t vecincout = svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

        {// stepout == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

            input = svsel_s32(ffttout,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(ffttout,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min

            const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

            input2 = svsel_s32(ffttout,
                              svsel_s32(mask2, input2Perm, input2),// max
                              svsel_s32(mask2, input2, input2Perm));// min

            input2_val = svsel_s32(ffttout,
                              svsel_s32(mask2, input2Perm_val, input2_val),// max
                              svsel_s32(mask2, input2_val, input2Perm_val));// min

            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

            input3 = svsel_s32(ffttout,
                              svsel_s32(mask3, input3Perm, input3),// max
                              svsel_s32(mask3, input3, input3Perm));// min

            input3_val = svsel_s32(ffttout,
                              svsel_s32(mask3, input3Perm_val, input3_val),// max
                              svsel_s32(mask3, input3_val, input3Perm_val));// min
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b32(ffttout,ffttout);
            vecincout = svsel_s32(ffttout,
                                      svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                                      svadd_n_s32_z(svptrue_b32(), vecincout, stepout));

            {
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(ffttout,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(ffttout,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(ffttout,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(ffttout,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(ffttout,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(ffttout,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min
            }

            svbool_t fftt = svuzp2_b32(ffttout,ffttout);

            svint32_t vecinc = svdup_s32(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min

                fftt = svuzp2_b32(fftt,fftt);
                vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min
            }
        }
    }
    {
        CoreExchange(input, input2,
                     input_val, input2_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
        }
    }
#endif
    {
        CoreExchange(input2, input3,
                     input2_val, input3_val);
    }
#ifdef NOOPTIM
    CoreSmallEnd2(input, input2,
                  input_val, input2_val);
    CoreSmallSortEnd(input3,
                 input3_val);
#else
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
        }
    }
#endif
}

inline void CoreSmallEnd4(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                          svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val){
    {
        CoreExchangeEnd(input, input3,
                        input_val, input3_val);
        CoreExchangeEnd(input2, input4,
                        input2_val, input4_val);
    }
#ifdef NOOPTIM
    CoreSmallEnd2(input, input2,
                   input_val, input2_val);
    CoreSmallEnd2(input3, input4,
                   input3_val, input4_val);
#else
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
    }
    {
        CoreExchangeEnd(input3, input4,
                        input3_val, input4_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
        }
    }
#endif
}

inline void CoreSmallSort4(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                           svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val){
#ifdef NOOPTIM
    CoreSmallSort2(input, input2,
                   input_val, input2_val);
    CoreSmallSort2(input3, input4,
                   input3_val, input4_val);
#else
    {
        const int nbValuesInVec = svcntw();

        const svint32_t vecindex = svindex_s32(0, 1);

        svbool_t ffttout = svzip1_b32(svpfalse_b(),svptrue_b32());

        svint32_t vecincout = svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

        {// stepout == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

            input = svsel_s32(ffttout,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(ffttout,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min

            const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

            input2 = svsel_s32(ffttout,
                              svsel_s32(mask2, input2Perm, input2),// max
                              svsel_s32(mask2, input2, input2Perm));// min

            input2_val = svsel_s32(ffttout,
                              svsel_s32(mask2, input2Perm_val, input2_val),// max
                              svsel_s32(mask2, input2_val, input2Perm_val));// min

            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

            input3 = svsel_s32(ffttout,
                              svsel_s32(mask3, input3Perm, input3),// max
                              svsel_s32(mask3, input3, input3Perm));// min

            input3_val = svsel_s32(ffttout,
                              svsel_s32(mask3, input3Perm_val, input3_val),// max
                              svsel_s32(mask3, input3_val, input3Perm_val));// min

            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

            input4 = svsel_s32(ffttout,
                              svsel_s32(mask4, input4Perm, input4),// max
                              svsel_s32(mask4, input4, input4Perm));// min

            input4_val = svsel_s32(ffttout,
                              svsel_s32(mask4, input4Perm_val, input4_val),// max
                              svsel_s32(mask4, input4_val, input4Perm_val));// min
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b32(ffttout,ffttout);
            vecincout = svsel_s32(ffttout,
                                      svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                                      svadd_n_s32_z(svptrue_b32(), vecincout, stepout));

            {
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(ffttout,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(ffttout,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(ffttout,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(ffttout,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(ffttout,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(ffttout,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min

                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

                input4 = svsel_s32(ffttout,
                                  svsel_s32(mask4, input4Perm, input4),// max
                                  svsel_s32(mask4, input4, input4Perm));// min

                input4_val = svsel_s32(ffttout,
                                  svsel_s32(mask4, input4Perm_val, input4_val),// max
                                  svsel_s32(mask4, input4_val, input4Perm_val));// min
            }

            svbool_t fftt = svuzp2_b32(ffttout,ffttout);

            svint32_t vecinc = svdup_s32(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min

                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

                input4 = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm, input4),// max
                                  svsel_s32(mask4, input4, input4Perm));// min

                input4_val = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm_val, input4_val),// max
                                  svsel_s32(mask4, input4_val, input4Perm_val));// min

                fftt = svuzp2_b32(fftt,fftt);
                vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min

                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

                input4 = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm, input4),// max
                                  svsel_s32(mask4, input4, input4Perm));// min

                input4_val = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm_val, input4_val),// max
                                  svsel_s32(mask4, input4_val, input4Perm_val));// min
            }
        }
    }
    {
        CoreExchange(input, input2,
                     input_val, input2_val);
    }
    {
        CoreExchange(input3, input4,
                     input3_val, input4_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min

            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm, input3),// max
                              svsel_s32(mask3, input3, input3Perm));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm_val, input3_val),// max
                              svsel_s32(mask3, input3_val, input3Perm_val));// min

            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm, input4),// max
                              svsel_s32(mask4, input4, input4Perm));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm_val, input4_val),// max
                              svsel_s32(mask4, input4_val, input4Perm_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min

            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm, input3),// max
                              svsel_s32(mask3, input3, input3Perm));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm_val, input3_val),// max
                              svsel_s32(mask3, input3_val, input3Perm_val));// min

            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm, input4),// max
                              svsel_s32(mask4, input4, input4Perm));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm_val, input4_val),// max
                              svsel_s32(mask4, input4_val, input4Perm_val));// min
        }
    }
#endif
    {
        CoreExchange(input, input4,
                     input_val, input4_val);
        CoreExchange(input2, input3,
                     input2_val, input3_val);
    }
#ifdef NOOPTIM
    CoreSmallEnd2(input, input2,
                  input_val, input2_val);
    CoreSmallEnd2(input3, input4,
                  input3_val, input4_val);
#else
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
    }
    {
        CoreExchangeEnd(input3, input4,
                        input3_val, input4_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
        }
    }
#endif
}

inline void CoreSmallEnd5(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4, svint32_t& input5,
                          svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                          svint32_t& input5_val){
    {
        CoreExchangeEnd(input, input5,
                        input_val, input5_val);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4,
                   input_val, input2_val, input3_val, input4_val);
    CoreSmallSortEnd(input5,
                  input5_val);
#else
    {
        CoreExchangeEnd(input, input3,
                        input_val, input3_val);
        CoreExchangeEnd(input2, input4,
                        input2_val, input4_val);
    }
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
        CoreExchangeEnd(input3, input4,
                        input3_val, input4_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);
            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, inputPerm5);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5, input5),// max
                              svsel_s32(mask5, input5, inputPerm5));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5_val, input5_val),// max
                              svsel_s32(mask5, input5_val, inputPerm5_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);
            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, inputPerm5);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5, input5),// max
                              svsel_s32(mask5, input5, inputPerm5));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5_val, input5_val),// max
                              svsel_s32(mask5, input5_val, inputPerm5_val));// min
        }
    }
#endif
}

inline void CoreSmallSort5(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                           svint32_t& input5,
                           svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                           svint32_t& input5_val){
#ifdef NOOPTIM
    CoreSmallSort4(input, input2, input3, input4,
                   input_val, input2_val, input3_val, input4_val);
    CoreSmallSort(input5,
                  input5_val);
#else
    {
        const int nbValuesInVec = svcntw();

        const svint32_t vecindex = svindex_s32(0, 1);

        svbool_t ffttout = svzip1_b32(svpfalse_b(),svptrue_b32());

        svint32_t vecincout = svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

        {// stepout == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

            input = svsel_s32(ffttout,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(ffttout,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min

            const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

            input2 = svsel_s32(ffttout,
                              svsel_s32(mask2, input2Perm, input2),// max
                              svsel_s32(mask2, input2, input2Perm));// min

            input2_val = svsel_s32(ffttout,
                              svsel_s32(mask2, input2Perm_val, input2_val),// max
                              svsel_s32(mask2, input2_val, input2Perm_val));// min

            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

            input3 = svsel_s32(ffttout,
                              svsel_s32(mask3, input3Perm, input3),// max
                              svsel_s32(mask3, input3, input3Perm));// min

            input3_val = svsel_s32(ffttout,
                              svsel_s32(mask3, input3Perm_val, input3_val),// max
                              svsel_s32(mask3, input3_val, input3Perm_val));// min

            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

            input4 = svsel_s32(ffttout,
                              svsel_s32(mask4, input4Perm, input4),// max
                              svsel_s32(mask4, input4, input4Perm));// min

            input4_val = svsel_s32(ffttout,
                              svsel_s32(mask4, input4Perm_val, input4_val),// max
                              svsel_s32(mask4, input4_val, input4Perm_val));// min

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

            input5 = svsel_s32(ffttout,
                              svsel_s32(mask5, input5Perm, input5),// max
                              svsel_s32(mask5, input5, input5Perm));// min

            input5_val = svsel_s32(ffttout,
                              svsel_s32(mask5, input5Perm_val, input5_val),// max
                              svsel_s32(mask5, input5_val, input5Perm_val));// min
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b32(ffttout,ffttout);
            vecincout = svsel_s32(ffttout,
                                      svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                                      svadd_n_s32_z(svptrue_b32(), vecincout, stepout));

            {
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(ffttout,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(ffttout,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(ffttout,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(ffttout,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(ffttout,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(ffttout,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min

                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

                input4 = svsel_s32(ffttout,
                                  svsel_s32(mask4, input4Perm, input4),// max
                                  svsel_s32(mask4, input4, input4Perm));// min

                input4_val = svsel_s32(ffttout,
                                  svsel_s32(mask4, input4Perm_val, input4_val),// max
                                  svsel_s32(mask4, input4_val, input4Perm_val));// min

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

                input5 = svsel_s32(ffttout,
                                  svsel_s32(mask5, input5Perm, input5),// max
                                  svsel_s32(mask5, input5, input5Perm));// min

                input5_val = svsel_s32(ffttout,
                                  svsel_s32(mask5, input5Perm_val, input5_val),// max
                                  svsel_s32(mask5, input5_val, input5Perm_val));// min
            }

            svbool_t fftt = svuzp2_b32(ffttout,ffttout);

            svint32_t vecinc = svdup_s32(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min

                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

                input4 = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm, input4),// max
                                  svsel_s32(mask4, input4, input4Perm));// min

                input4_val = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm_val, input4_val),// max
                                  svsel_s32(mask4, input4_val, input4Perm_val));// min

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

                input5 = svsel_s32(fftt,
                                  svsel_s32(mask5, input5Perm, input5),// max
                                  svsel_s32(mask5, input5, input5Perm));// min

                input5_val = svsel_s32(fftt,
                                  svsel_s32(mask5, input5Perm_val, input5_val),// max
                                  svsel_s32(mask5, input5_val, input5Perm_val));// min

                fftt = svuzp2_b32(fftt,fftt);
                vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min

                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

                input4 = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm, input4),// max
                                  svsel_s32(mask4, input4, input4Perm));// min

                input4_val = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm_val, input4_val),// max
                                  svsel_s32(mask4, input4_val, input4Perm_val));// min

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

                input5 = svsel_s32(fftt,
                                  svsel_s32(mask5, input5Perm, input5),// max
                                  svsel_s32(mask5, input5, input5Perm));// min

                input5_val = svsel_s32(fftt,
                                  svsel_s32(mask5, input5Perm_val, input5_val),// max
                                  svsel_s32(mask5, input5_val, input5Perm_val));// min
            }
        }
    }
    {
        CoreExchange(input, input2,
                     input_val, input2_val);
    }
    {
        CoreExchange(input3, input4,
                     input3_val, input4_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min

            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm, input3),// max
                              svsel_s32(mask3, input3, input3Perm));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm_val, input3_val),// max
                              svsel_s32(mask3, input3_val, input3Perm_val));// min

            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm, input4),// max
                              svsel_s32(mask4, input4, input4Perm));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm_val, input4_val),// max
                              svsel_s32(mask4, input4_val, input4Perm_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min

            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm, input3),// max
                              svsel_s32(mask3, input3, input3Perm));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm_val, input3_val),// max
                              svsel_s32(mask3, input3_val, input3Perm_val));// min

            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm, input4),// max
                              svsel_s32(mask4, input4, input4Perm));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm_val, input4_val),// max
                              svsel_s32(mask4, input4_val, input4Perm_val));// min
        }
    }
    {
        CoreExchange(input, input4,
                     input_val, input4_val);
        CoreExchange(input2, input3,
                     input2_val, input3_val);
    }
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
    }
    {
        CoreExchangeEnd(input3, input4,
                        input3_val, input4_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
        }
    }
#endif
    {
        CoreExchange(input4, input5,
                     input4_val, input5_val);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4,
                  input_val, input2_val, input3_val, input4_val);
    CoreSmallSortEnd(input5,
                 input5_val);
#else
    {
        CoreExchangeEnd(input, input3,
                        input_val, input3_val);
        CoreExchangeEnd(input2, input4,
                        input2_val, input4_val);
    }
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
        CoreExchangeEnd(input3, input4,
                        input3_val, input4_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);
            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, inputPerm5);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5, input5),// max
                              svsel_s32(mask5, input5, inputPerm5));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5_val, input5_val),// max
                              svsel_s32(mask5, input5_val, inputPerm5_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);
            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, inputPerm5);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5, input5),// max
                              svsel_s32(mask5, input5, inputPerm5));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5_val, input5_val),// max
                              svsel_s32(mask5, input5_val, inputPerm5_val));// min
        }
    }
#endif
}

inline void CoreSmallEnd6(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                          svint32_t& input5, svint32_t& input6,
                          svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                          svint32_t& input5_val, svint32_t& input6_val){
    {
        CoreExchangeEnd(input, input5,
                        input_val, input5_val);
        CoreExchangeEnd(input2, input6,
                        input2_val, input6_val);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4,
                   input_val, input2_val, input3_val, input4_val);
    CoreSmallEnd2(input5, input6,
                   input5_val, input6_val);
#else
    {
        CoreExchangeEnd(input, input3,
                        input_val, input3_val);
        CoreExchangeEnd(input2, input4,
                        input2_val, input4_val);
    }
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
        CoreExchangeEnd(input3, input4,
                        input3_val, input4_val);
    }
    {
        CoreExchangeEnd(input5, input6,
                        input5_val, input6_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);
            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, inputPerm5);
            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, inputPerm6);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5, input5),// max
                              svsel_s32(mask5, input5, inputPerm5));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5_val, input5_val),// max
                              svsel_s32(mask5, input5_val, inputPerm5_val));// min
            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6, input6),// max
                              svsel_s32(mask6, input6, inputPerm6));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6_val, input6_val),// max
                              svsel_s32(mask6, input6_val, inputPerm6_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);
            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, inputPerm5);
            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, inputPerm6);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5, input5),// max
                              svsel_s32(mask5, input5, inputPerm5));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5_val, input5_val),// max
                              svsel_s32(mask5, input5_val, inputPerm5_val));// min
            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6, input6),// max
                              svsel_s32(mask6, input6, inputPerm6));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6_val, input6_val),// max
                              svsel_s32(mask6, input6_val, inputPerm6_val));// min
        }
    }
#endif
}

inline void CoreSmallSort6(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                           svint32_t& input5, svint32_t& input6,
                           svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                           svint32_t& input5_val, svint32_t& input6_val){
#ifdef NOOPTIM
    CoreSmallSort4(input, input2, input3, input4,
                   input_val, input2_val, input3_val, input4_val);
    CoreSmallSort2(input5, input6,
                   input5_val, input6_val);
#else
    {
        const int nbValuesInVec = svcntw();

        const svint32_t vecindex = svindex_s32(0, 1);

        svbool_t ffttout = svzip1_b32(svpfalse_b(),svptrue_b32());

        svint32_t vecincout = svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

        {// stepout == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

            input = svsel_s32(ffttout,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(ffttout,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min

            const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

            input2 = svsel_s32(ffttout,
                              svsel_s32(mask2, input2Perm, input2),// max
                              svsel_s32(mask2, input2, input2Perm));// min

            input2_val = svsel_s32(ffttout,
                              svsel_s32(mask2, input2Perm_val, input2_val),// max
                              svsel_s32(mask2, input2_val, input2Perm_val));// min

            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

            input3 = svsel_s32(ffttout,
                              svsel_s32(mask3, input3Perm, input3),// max
                              svsel_s32(mask3, input3, input3Perm));// min

            input3_val = svsel_s32(ffttout,
                              svsel_s32(mask3, input3Perm_val, input3_val),// max
                              svsel_s32(mask3, input3_val, input3Perm_val));// min

            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

            input4 = svsel_s32(ffttout,
                              svsel_s32(mask4, input4Perm, input4),// max
                              svsel_s32(mask4, input4, input4Perm));// min

            input4_val = svsel_s32(ffttout,
                              svsel_s32(mask4, input4Perm_val, input4_val),// max
                              svsel_s32(mask4, input4_val, input4Perm_val));// min

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

            input5 = svsel_s32(ffttout,
                              svsel_s32(mask5, input5Perm, input5),// max
                              svsel_s32(mask5, input5, input5Perm));// min

            input5_val = svsel_s32(ffttout,
                              svsel_s32(mask5, input5Perm_val, input5_val),// max
                              svsel_s32(mask5, input5_val, input5Perm_val));// min

            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

            input6 = svsel_s32(ffttout,
                              svsel_s32(mask6, input6Perm, input6),// max
                              svsel_s32(mask6, input6, input6Perm));// min

            input6_val = svsel_s32(ffttout,
                              svsel_s32(mask6, input6Perm_val, input6_val),// max
                              svsel_s32(mask6, input6_val, input6Perm_val));// min
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b32(ffttout,ffttout);
            vecincout = svsel_s32(ffttout,
                                      svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                                      svadd_n_s32_z(svptrue_b32(), vecincout, stepout));

            {
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(ffttout,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(ffttout,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(ffttout,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(ffttout,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(ffttout,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(ffttout,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min

                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

                input4 = svsel_s32(ffttout,
                                  svsel_s32(mask4, input4Perm, input4),// max
                                  svsel_s32(mask4, input4, input4Perm));// min

                input4_val = svsel_s32(ffttout,
                                  svsel_s32(mask4, input4Perm_val, input4_val),// max
                                  svsel_s32(mask4, input4_val, input4Perm_val));// min

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

                input5 = svsel_s32(ffttout,
                                  svsel_s32(mask5, input5Perm, input5),// max
                                  svsel_s32(mask5, input5, input5Perm));// min

                input5_val = svsel_s32(ffttout,
                                  svsel_s32(mask5, input5Perm_val, input5_val),// max
                                  svsel_s32(mask5, input5_val, input5Perm_val));// min

                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

                input6 = svsel_s32(ffttout,
                                  svsel_s32(mask6, input6Perm, input6),// max
                                  svsel_s32(mask6, input6, input6Perm));// min

                input6_val = svsel_s32(ffttout,
                                  svsel_s32(mask6, input6Perm_val, input6_val),// max
                                  svsel_s32(mask6, input6_val, input6Perm_val));// min
            }

            svbool_t fftt = svuzp2_b32(ffttout,ffttout);

            svint32_t vecinc = svdup_s32(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min

                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

                input4 = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm, input4),// max
                                  svsel_s32(mask4, input4, input4Perm));// min

                input4_val = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm_val, input4_val),// max
                                  svsel_s32(mask4, input4_val, input4Perm_val));// min

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

                input5 = svsel_s32(fftt,
                                  svsel_s32(mask5, input5Perm, input5),// max
                                  svsel_s32(mask5, input5, input5Perm));// min

                input5_val = svsel_s32(fftt,
                                  svsel_s32(mask5, input5Perm_val, input5_val),// max
                                  svsel_s32(mask5, input5_val, input5Perm_val));// min

                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

                input6 = svsel_s32(fftt,
                                  svsel_s32(mask6, input6Perm, input6),// max
                                  svsel_s32(mask6, input6, input6Perm));// min

                input6_val = svsel_s32(fftt,
                                  svsel_s32(mask6, input6Perm_val, input6_val),// max
                                  svsel_s32(mask6, input6_val, input6Perm_val));// min

                fftt = svuzp2_b32(fftt,fftt);
                vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min

                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

                input4 = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm, input4),// max
                                  svsel_s32(mask4, input4, input4Perm));// min

                input4_val = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm_val, input4_val),// max
                                  svsel_s32(mask4, input4_val, input4Perm_val));// min

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

                input5 = svsel_s32(fftt,
                                  svsel_s32(mask5, input5Perm, input5),// max
                                  svsel_s32(mask5, input5, input5Perm));// min

                input5_val = svsel_s32(fftt,
                                  svsel_s32(mask5, input5Perm_val, input5_val),// max
                                  svsel_s32(mask5, input5_val, input5Perm_val));// min

                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

                input6 = svsel_s32(fftt,
                                  svsel_s32(mask6, input6Perm, input6),// max
                                  svsel_s32(mask6, input6, input6Perm));// min

                input6_val = svsel_s32(fftt,
                                  svsel_s32(mask6, input6Perm_val, input6_val),// max
                                  svsel_s32(mask6, input6_val, input6Perm_val));// min
            }
        }
    }
    {
        CoreExchange(input, input2,
                     input_val, input2_val);
    }
    {
        CoreExchange(input3, input4,
                     input3_val, input4_val);
    }
    {
        CoreExchange(input5, input6,
                     input5_val, input6_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min

            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm, input3),// max
                              svsel_s32(mask3, input3, input3Perm));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm_val, input3_val),// max
                              svsel_s32(mask3, input3_val, input3Perm_val));// min

            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm, input4),// max
                              svsel_s32(mask4, input4, input4Perm));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm_val, input4_val),// max
                              svsel_s32(mask4, input4_val, input4Perm_val));// min

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm, input5),// max
                              svsel_s32(mask5, input5, input5Perm));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm_val, input5_val),// max
                              svsel_s32(mask5, input5_val, input5Perm_val));// min

            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm, input6),// max
                              svsel_s32(mask6, input6, input6Perm));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm_val, input6_val),// max
                              svsel_s32(mask6, input6_val, input6Perm_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min

            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm, input3),// max
                              svsel_s32(mask3, input3, input3Perm));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm_val, input3_val),// max
                              svsel_s32(mask3, input3_val, input3Perm_val));// min

            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm, input4),// max
                              svsel_s32(mask4, input4, input4Perm));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm_val, input4_val),// max
                              svsel_s32(mask4, input4_val, input4Perm_val));// min

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm, input5),// max
                              svsel_s32(mask5, input5, input5Perm));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm_val, input5_val),// max
                              svsel_s32(mask5, input5_val, input5Perm_val));// min

            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm, input6),// max
                              svsel_s32(mask6, input6, input6Perm));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm_val, input6_val),// max
                              svsel_s32(mask6, input6_val, input6Perm_val));// min
        }
    }
    {
        CoreExchange(input, input4,
                     input_val, input4_val);
        CoreExchange(input2, input3,
                     input2_val, input3_val);
    }
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
    }
    {
        CoreExchangeEnd(input3, input4,
                        input3_val, input4_val);
    }
    {
        CoreExchangeEnd(input5, input6,
                        input5_val, input6_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm, input5),// max
                              svsel_s32(mask5, input5, input5Perm));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm_val, input5_val),// max
                              svsel_s32(mask5, input5_val, input5Perm_val));// min

            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm, input6),// max
                              svsel_s32(mask6, input6, input6Perm));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm_val, input6_val),// max
                              svsel_s32(mask6, input6_val, input6Perm_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm, input5),// max
                              svsel_s32(mask5, input5, input5Perm));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm_val, input5_val),// max
                              svsel_s32(mask5, input5_val, input5Perm_val));// min

            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm, input6),// max
                              svsel_s32(mask6, input6, input6Perm));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm_val, input6_val),// max
                              svsel_s32(mask6, input6_val, input6Perm_val));// min
        }
    }
#endif
    {
        CoreExchange(input3, input6,
                     input3_val, input6_val);
        CoreExchange(input4, input5,
                     input4_val, input5_val);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4,
                  input_val, input2_val, input3_val, input4_val);
    CoreSmallEnd2(input5, input6,
                  input5_val, input6_val);
#else
    {
        CoreExchangeEnd(input, input3,
                        input_val, input3_val);
        CoreExchangeEnd(input2, input4,
                        input2_val, input4_val);
    }
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
        CoreExchangeEnd(input3, input4,
                        input3_val, input4_val);
    }
    {
        CoreExchangeEnd(input5, input6,
                        input5_val, input6_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);
            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, inputPerm5);
            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, inputPerm6);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5, input5),// max
                              svsel_s32(mask5, input5, inputPerm5));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5_val, input5_val),// max
                              svsel_s32(mask5, input5_val, inputPerm5_val));// min
            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6, input6),// max
                              svsel_s32(mask6, input6, inputPerm6));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6_val, input6_val),// max
                              svsel_s32(mask6, input6_val, inputPerm6_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);
            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, inputPerm5);
            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, inputPerm6);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5, input5),// max
                              svsel_s32(mask5, input5, inputPerm5));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5_val, input5_val),// max
                              svsel_s32(mask5, input5_val, inputPerm5_val));// min
            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6, input6),// max
                              svsel_s32(mask6, input6, inputPerm6));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6_val, input6_val),// max
                              svsel_s32(mask6, input6_val, inputPerm6_val));// min
        }
    }
#endif
}

inline void CoreSmallEnd7(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                          svint32_t& input5, svint32_t& input6, svint32_t& input7,
                          svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                          svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val){
    {
        CoreExchangeEnd(input, input5,
                        input_val, input5_val);
        CoreExchangeEnd(input2, input6,
                        input2_val, input6_val);
        CoreExchangeEnd(input3, input7,
                        input3_val, input7_val);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4,
                   input_val, input2_val, input3_val, input4_val);
    CoreSmallEnd3(input5, input6, input7,
                   input5_val, input6_val, input7_val);
#else
    {
        CoreExchangeEnd(input, input3,
                        input_val, input3_val);
        CoreExchangeEnd(input2, input4,
                        input2_val, input4_val);
    }
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
        CoreExchangeEnd(input3, input4,
                        input3_val, input4_val);
    }
    {
        CoreExchangeEnd(input5, input7,
                        input5_val, input7_val);
    }
    {
        CoreExchangeEnd(input5, input6,
                        input5_val, input6_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7 = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);
            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, inputPerm5);
            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, inputPerm6);
            svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, inputPerm7);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5, input5),// max
                              svsel_s32(mask5, input5, inputPerm5));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5_val, input5_val),// max
                              svsel_s32(mask5, input5_val, inputPerm5_val));// min
            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6, input6),// max
                              svsel_s32(mask6, input6, inputPerm6));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6_val, input6_val),// max
                              svsel_s32(mask6, input6_val, inputPerm6_val));// min
            input7 = svsel_s32(fftt,
                              svsel_s32(mask7, inputPerm7, input7),// max
                              svsel_s32(mask7, input7, inputPerm7));// min

            input7_val = svsel_s32(fftt,
                              svsel_s32(mask7, inputPerm7_val, input7_val),// max
                              svsel_s32(mask7, input7_val, inputPerm7_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7 = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);
            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, inputPerm5);
            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, inputPerm6);
            svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, inputPerm7);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5, input5),// max
                              svsel_s32(mask5, input5, inputPerm5));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5_val, input5_val),// max
                              svsel_s32(mask5, input5_val, inputPerm5_val));// min
            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6, input6),// max
                              svsel_s32(mask6, input6, inputPerm6));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6_val, input6_val),// max
                              svsel_s32(mask6, input6_val, inputPerm6_val));// min
            input7 = svsel_s32(fftt,
                              svsel_s32(mask7, inputPerm7, input7),// max
                              svsel_s32(mask7, input7, inputPerm7));// min

            input7_val = svsel_s32(fftt,
                              svsel_s32(mask7, inputPerm7_val, input7_val),// max
                              svsel_s32(mask7, input7_val, inputPerm7_val));// min
        }
    }
#endif
}

inline void CoreSmallSort7(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                           svint32_t& input5, svint32_t& input6, svint32_t& input7,
                           svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                           svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val){
#ifdef NOOPTIM
    CoreSmallSort4(input, input2, input3, input4,
                   input_val, input2_val, input3_val, input4_val);
    CoreSmallSort3(input5, input6, input7,
                   input5_val, input6_val, input7_val);
#else
    {
        const int nbValuesInVec = svcntw();

        const svint32_t vecindex = svindex_s32(0, 1);

        svbool_t ffttout = svzip1_b32(svpfalse_b(),svptrue_b32());

        svint32_t vecincout = svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

        {// stepout == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

            input = svsel_s32(ffttout,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(ffttout,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min

            const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

            input2 = svsel_s32(ffttout,
                              svsel_s32(mask2, input2Perm, input2),// max
                              svsel_s32(mask2, input2, input2Perm));// min

            input2_val = svsel_s32(ffttout,
                              svsel_s32(mask2, input2Perm_val, input2_val),// max
                              svsel_s32(mask2, input2_val, input2Perm_val));// min

            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

            input3 = svsel_s32(ffttout,
                              svsel_s32(mask3, input3Perm, input3),// max
                              svsel_s32(mask3, input3, input3Perm));// min

            input3_val = svsel_s32(ffttout,
                              svsel_s32(mask3, input3Perm_val, input3_val),// max
                              svsel_s32(mask3, input3_val, input3Perm_val));// min

            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

            input4 = svsel_s32(ffttout,
                              svsel_s32(mask4, input4Perm, input4),// max
                              svsel_s32(mask4, input4, input4Perm));// min

            input4_val = svsel_s32(ffttout,
                              svsel_s32(mask4, input4Perm_val, input4_val),// max
                              svsel_s32(mask4, input4_val, input4Perm_val));// min

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

            input5 = svsel_s32(ffttout,
                              svsel_s32(mask5, input5Perm, input5),// max
                              svsel_s32(mask5, input5, input5Perm));// min

            input5_val = svsel_s32(ffttout,
                              svsel_s32(mask5, input5Perm_val, input5_val),// max
                              svsel_s32(mask5, input5_val, input5Perm_val));// min

            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

            input6 = svsel_s32(ffttout,
                              svsel_s32(mask6, input6Perm, input6),// max
                              svsel_s32(mask6, input6, input6Perm));// min

            input6_val = svsel_s32(ffttout,
                              svsel_s32(mask6, input6Perm_val, input6_val),// max
                              svsel_s32(mask6, input6_val, input6Perm_val));// min

            const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t input7Perm_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, input7Perm);

            input7 = svsel_s32(ffttout,
                              svsel_s32(mask7, input7Perm, input7),// max
                              svsel_s32(mask7, input7, input7Perm));// min

            input7_val = svsel_s32(ffttout,
                              svsel_s32(mask7, input7Perm_val, input7_val),// max
                              svsel_s32(mask7, input7_val, input7Perm_val));// min
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b32(ffttout,ffttout);
            vecincout = svsel_s32(ffttout,
                                      svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                                      svadd_n_s32_z(svptrue_b32(), vecincout, stepout));

            {
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(ffttout,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(ffttout,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(ffttout,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(ffttout,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(ffttout,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(ffttout,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min

                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

                input4 = svsel_s32(ffttout,
                                  svsel_s32(mask4, input4Perm, input4),// max
                                  svsel_s32(mask4, input4, input4Perm));// min

                input4_val = svsel_s32(ffttout,
                                  svsel_s32(mask4, input4Perm_val, input4_val),// max
                                  svsel_s32(mask4, input4_val, input4Perm_val));// min

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

                input5 = svsel_s32(ffttout,
                                  svsel_s32(mask5, input5Perm, input5),// max
                                  svsel_s32(mask5, input5, input5Perm));// min

                input5_val = svsel_s32(ffttout,
                                  svsel_s32(mask5, input5Perm_val, input5_val),// max
                                  svsel_s32(mask5, input5_val, input5Perm_val));// min

                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

                input6 = svsel_s32(ffttout,
                                  svsel_s32(mask6, input6Perm, input6),// max
                                  svsel_s32(mask6, input6, input6Perm));// min

                input6_val = svsel_s32(ffttout,
                                  svsel_s32(mask6, input6Perm_val, input6_val),// max
                                  svsel_s32(mask6, input6_val, input6Perm_val));// min

                const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
                const svint32_t input7Perm_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, input7Perm);

                input7 = svsel_s32(ffttout,
                                  svsel_s32(mask7, input7Perm, input7),// max
                                  svsel_s32(mask7, input7, input7Perm));// min

                input7_val = svsel_s32(ffttout,
                                  svsel_s32(mask7, input7Perm_val, input7_val),// max
                                  svsel_s32(mask7, input7_val, input7Perm_val));// min
            }

            svbool_t fftt = svuzp2_b32(ffttout,ffttout);

            svint32_t vecinc = svdup_s32(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min

                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

                input4 = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm, input4),// max
                                  svsel_s32(mask4, input4, input4Perm));// min

                input4_val = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm_val, input4_val),// max
                                  svsel_s32(mask4, input4_val, input4Perm_val));// min

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

                input5 = svsel_s32(fftt,
                                  svsel_s32(mask5, input5Perm, input5),// max
                                  svsel_s32(mask5, input5, input5Perm));// min

                input5_val = svsel_s32(fftt,
                                  svsel_s32(mask5, input5Perm_val, input5_val),// max
                                  svsel_s32(mask5, input5_val, input5Perm_val));// min

                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

                input6 = svsel_s32(fftt,
                                  svsel_s32(mask6, input6Perm, input6),// max
                                  svsel_s32(mask6, input6, input6Perm));// min

                input6_val = svsel_s32(fftt,
                                  svsel_s32(mask6, input6Perm_val, input6_val),// max
                                  svsel_s32(mask6, input6_val, input6Perm_val));// min

                const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
                const svint32_t input7Perm_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, input7Perm);

                input7 = svsel_s32(fftt,
                                  svsel_s32(mask7, input7Perm, input7),// max
                                  svsel_s32(mask7, input7, input7Perm));// min

                input7_val = svsel_s32(fftt,
                                  svsel_s32(mask7, input7Perm_val, input7_val),// max
                                  svsel_s32(mask7, input7_val, input7Perm_val));// min

                fftt = svuzp2_b32(fftt,fftt);
                vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min

                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

                input4 = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm, input4),// max
                                  svsel_s32(mask4, input4, input4Perm));// min

                input4_val = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm_val, input4_val),// max
                                  svsel_s32(mask4, input4_val, input4Perm_val));// min

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

                input5 = svsel_s32(fftt,
                                  svsel_s32(mask5, input5Perm, input5),// max
                                  svsel_s32(mask5, input5, input5Perm));// min

                input5_val = svsel_s32(fftt,
                                  svsel_s32(mask5, input5Perm_val, input5_val),// max
                                  svsel_s32(mask5, input5_val, input5Perm_val));// min

                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

                input6 = svsel_s32(fftt,
                                  svsel_s32(mask6, input6Perm, input6),// max
                                  svsel_s32(mask6, input6, input6Perm));// min

                input6_val = svsel_s32(fftt,
                                  svsel_s32(mask6, input6Perm_val, input6_val),// max
                                  svsel_s32(mask6, input6_val, input6Perm_val));// min

                const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
                const svint32_t input7Perm_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, input7Perm);

                input7 = svsel_s32(fftt,
                                  svsel_s32(mask7, input7Perm, input7),// max
                                  svsel_s32(mask7, input7, input7Perm));// min

                input7_val = svsel_s32(fftt,
                                  svsel_s32(mask7, input7Perm_val, input7_val),// max
                                  svsel_s32(mask7, input7_val, input7Perm_val));// min
            }
        }
    }
    {
        CoreExchange(input, input2,
                     input_val, input2_val);
    }
    {
        CoreExchange(input3, input4,
                     input3_val, input4_val);
    }
    {
        CoreExchange(input5, input6,
                     input5_val, input6_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min

            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm, input3),// max
                              svsel_s32(mask3, input3, input3Perm));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm_val, input3_val),// max
                              svsel_s32(mask3, input3_val, input3Perm_val));// min

            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm, input4),// max
                              svsel_s32(mask4, input4, input4Perm));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm_val, input4_val),// max
                              svsel_s32(mask4, input4_val, input4Perm_val));// min

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm, input5),// max
                              svsel_s32(mask5, input5, input5Perm));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm_val, input5_val),// max
                              svsel_s32(mask5, input5_val, input5Perm_val));// min

            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm, input6),// max
                              svsel_s32(mask6, input6, input6Perm));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm_val, input6_val),// max
                              svsel_s32(mask6, input6_val, input6Perm_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min

            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm, input3),// max
                              svsel_s32(mask3, input3, input3Perm));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm_val, input3_val),// max
                              svsel_s32(mask3, input3_val, input3Perm_val));// min

            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm, input4),// max
                              svsel_s32(mask4, input4, input4Perm));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm_val, input4_val),// max
                              svsel_s32(mask4, input4_val, input4Perm_val));// min

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm, input5),// max
                              svsel_s32(mask5, input5, input5Perm));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm_val, input5_val),// max
                              svsel_s32(mask5, input5_val, input5Perm_val));// min

            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm, input6),// max
                              svsel_s32(mask6, input6, input6Perm));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm_val, input6_val),// max
                              svsel_s32(mask6, input6_val, input6Perm_val));// min
        }
    }
    {
        CoreExchange(input, input4,
                     input_val, input4_val);
        CoreExchange(input2, input3,
                     input2_val, input3_val);
    }
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
    }
    {
        CoreExchangeEnd(input3, input4,
                        input3_val, input4_val);
    }
    {
        CoreExchange(input6, input7,
                     input6_val, input7_val);
    }
    {
        CoreExchangeEnd(input5, input6,
                        input5_val, input6_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm, input5),// max
                              svsel_s32(mask5, input5, input5Perm));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm_val, input5_val),// max
                              svsel_s32(mask5, input5_val, input5Perm_val));// min

            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm, input6),// max
                              svsel_s32(mask6, input6, input6Perm));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm_val, input6_val),// max
                              svsel_s32(mask6, input6_val, input6Perm_val));// min

            const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t input7Perm_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, input7Perm);

            input7 = svsel_s32(fftt,
                              svsel_s32(mask7, input7Perm, input7),// max
                              svsel_s32(mask7, input7, input7Perm));// min

            input7_val = svsel_s32(fftt,
                              svsel_s32(mask7, input7Perm_val, input7_val),// max
                              svsel_s32(mask7, input7_val, input7Perm_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm, input5),// max
                              svsel_s32(mask5, input5, input5Perm));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm_val, input5_val),// max
                              svsel_s32(mask5, input5_val, input5Perm_val));// min

            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm, input6),// max
                              svsel_s32(mask6, input6, input6Perm));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm_val, input6_val),// max
                              svsel_s32(mask6, input6_val, input6Perm_val));// min

            const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t input7Perm_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, input7Perm);

            input7 = svsel_s32(fftt,
                              svsel_s32(mask7, input7Perm, input7),// max
                              svsel_s32(mask7, input7, input7Perm));// min

            input7_val = svsel_s32(fftt,
                              svsel_s32(mask7, input7Perm_val, input7_val),// max
                              svsel_s32(mask7, input7_val, input7Perm_val));// min
        }
    }
#endif
    {
        CoreExchange(input2, input7,
                     input2_val, input7_val);
        CoreExchange(input3, input6,
                     input3_val, input6_val);
        CoreExchange(input4, input5,
                     input4_val, input5_val);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4,
                  input_val, input2_val, input3_val, input4_val);
    CoreSmallEnd3(input5, input6, input7,
                  input5_val, input6_val, input7_val);
#else
    {
        CoreExchangeEnd(input, input3,
                        input_val, input3_val);
        CoreExchangeEnd(input2, input4,
                        input2_val, input4_val);
    }
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
        CoreExchangeEnd(input3, input4,
                        input3_val, input4_val);
    }
    {
        CoreExchangeEnd(input5, input7,
                        input5_val, input7_val);
    }
    {
        CoreExchangeEnd(input5, input6,
                        input5_val, input6_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7 = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);
            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, inputPerm5);
            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, inputPerm6);
            svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, inputPerm7);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5, input5),// max
                              svsel_s32(mask5, input5, inputPerm5));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5_val, input5_val),// max
                              svsel_s32(mask5, input5_val, inputPerm5_val));// min
            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6, input6),// max
                              svsel_s32(mask6, input6, inputPerm6));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6_val, input6_val),// max
                              svsel_s32(mask6, input6_val, inputPerm6_val));// min
            input7 = svsel_s32(fftt,
                              svsel_s32(mask7, inputPerm7, input7),// max
                              svsel_s32(mask7, input7, inputPerm7));// min

            input7_val = svsel_s32(fftt,
                              svsel_s32(mask7, inputPerm7_val, input7_val),// max
                              svsel_s32(mask7, input7_val, inputPerm7_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7 = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);
            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, inputPerm5);
            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, inputPerm6);
            svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, inputPerm7);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5, input5),// max
                              svsel_s32(mask5, input5, inputPerm5));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5_val, input5_val),// max
                              svsel_s32(mask5, input5_val, inputPerm5_val));// min
            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6, input6),// max
                              svsel_s32(mask6, input6, inputPerm6));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6_val, input6_val),// max
                              svsel_s32(mask6, input6_val, inputPerm6_val));// min
            input7 = svsel_s32(fftt,
                              svsel_s32(mask7, inputPerm7, input7),// max
                              svsel_s32(mask7, input7, inputPerm7));// min

            input7_val = svsel_s32(fftt,
                              svsel_s32(mask7, inputPerm7_val, input7_val),// max
                              svsel_s32(mask7, input7_val, inputPerm7_val));// min
        }
    }
#endif
}


inline void CoreSmallEnd8(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                          svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                          svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                          svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val){
    {
        CoreExchangeEnd(input, input5,
                        input_val, input5_val);
        CoreExchangeEnd(input2, input6,
                        input2_val, input6_val);
        CoreExchangeEnd(input3, input7,
                        input3_val, input7_val);
        CoreExchangeEnd(input4, input8,
                        input4_val, input8_val);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4,
                   input_val, input2_val, input3_val, input4_val);
    CoreSmallEnd4(input5, input6, input7, input8,
                   input5_val, input6_val, input7_val, input8_val);
#else
    {
        CoreExchangeEnd(input, input3,
                        input_val, input3_val);
        CoreExchangeEnd(input2, input4,
                        input2_val, input4_val);
    }
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
        CoreExchangeEnd(input3, input4,
                        input3_val, input4_val);
    }
    {
        CoreExchangeEnd(input5, input7,
                        input5_val, input7_val);
        CoreExchangeEnd(input6, input8,
                        input6_val, input8_val);
    }
    {
        CoreExchangeEnd(input5, input6,
                        input5_val, input6_val);
        CoreExchangeEnd(input7, input8,
                        input7_val, input8_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7 = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm8 = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm8_val = svtbl_s32(input8_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);
            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, inputPerm5);
            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, inputPerm6);
            svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, inputPerm7);
            svbool_t mask8 = svcmplt_s32(svptrue_b32(), input8, inputPerm8);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5, input5),// max
                              svsel_s32(mask5, input5, inputPerm5));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5_val, input5_val),// max
                              svsel_s32(mask5, input5_val, inputPerm5_val));// min
            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6, input6),// max
                              svsel_s32(mask6, input6, inputPerm6));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6_val, input6_val),// max
                              svsel_s32(mask6, input6_val, inputPerm6_val));// min
            input7 = svsel_s32(fftt,
                              svsel_s32(mask7, inputPerm7, input7),// max
                              svsel_s32(mask7, input7, inputPerm7));// min

            input7_val = svsel_s32(fftt,
                              svsel_s32(mask7, inputPerm7_val, input7_val),// max
                              svsel_s32(mask7, input7_val, inputPerm7_val));// min
            input8 = svsel_s32(fftt,
                              svsel_s32(mask8, inputPerm8, input8),// max
                              svsel_s32(mask8, input8, inputPerm8));// min

            input8_val = svsel_s32(fftt,
                              svsel_s32(mask8, inputPerm8_val, input8_val),// max
                              svsel_s32(mask8, input8_val, inputPerm8_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7 = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm8 = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm8_val = svtbl_s32(input8_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);
            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, inputPerm5);
            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, inputPerm6);
            svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, inputPerm7);
            svbool_t mask8 = svcmplt_s32(svptrue_b32(), input8, inputPerm8);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5, input5),// max
                              svsel_s32(mask5, input5, inputPerm5));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5_val, input5_val),// max
                              svsel_s32(mask5, input5_val, inputPerm5_val));// min
            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6, input6),// max
                              svsel_s32(mask6, input6, inputPerm6));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6_val, input6_val),// max
                              svsel_s32(mask6, input6_val, inputPerm6_val));// min
            input7 = svsel_s32(fftt,
                              svsel_s32(mask7, inputPerm7, input7),// max
                              svsel_s32(mask7, input7, inputPerm7));// min

            input7_val = svsel_s32(fftt,
                              svsel_s32(mask7, inputPerm7_val, input7_val),// max
                              svsel_s32(mask7, input7_val, inputPerm7_val));// min
            input8 = svsel_s32(fftt,
                              svsel_s32(mask8, inputPerm8, input8),// max
                              svsel_s32(mask8, input8, inputPerm8));// min

            input8_val = svsel_s32(fftt,
                              svsel_s32(mask8, inputPerm8_val, input8_val),// max
                              svsel_s32(mask8, input8_val, inputPerm8_val));// min
        }
    }
#endif
}

inline void CoreSmallSort8(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                           svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                           svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                           svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val){
#ifdef NOOPTIM
    CoreSmallSort4(input, input2, input3, input4,
                   input_val, input2_val, input3_val, input4_val);
    CoreSmallSort4(input5, input6, input7, input8,
                   input5_val, input6_val, input7_val, input8_val);
#else
    {
        const int nbValuesInVec = svcntw();

        const svint32_t vecindex = svindex_s32(0, 1);

        svbool_t ffttout = svzip1_b32(svpfalse_b(),svptrue_b32());

        svint32_t vecincout = svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

        {// stepout == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

            input = svsel_s32(ffttout,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(ffttout,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min

            const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

            input2 = svsel_s32(ffttout,
                              svsel_s32(mask2, input2Perm, input2),// max
                              svsel_s32(mask2, input2, input2Perm));// min

            input2_val = svsel_s32(ffttout,
                              svsel_s32(mask2, input2Perm_val, input2_val),// max
                              svsel_s32(mask2, input2_val, input2Perm_val));// min

            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

            input3 = svsel_s32(ffttout,
                              svsel_s32(mask3, input3Perm, input3),// max
                              svsel_s32(mask3, input3, input3Perm));// min

            input3_val = svsel_s32(ffttout,
                              svsel_s32(mask3, input3Perm_val, input3_val),// max
                              svsel_s32(mask3, input3_val, input3Perm_val));// min

            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

            input4 = svsel_s32(ffttout,
                              svsel_s32(mask4, input4Perm, input4),// max
                              svsel_s32(mask4, input4, input4Perm));// min

            input4_val = svsel_s32(ffttout,
                              svsel_s32(mask4, input4Perm_val, input4_val),// max
                              svsel_s32(mask4, input4_val, input4Perm_val));// min

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

            input5 = svsel_s32(ffttout,
                              svsel_s32(mask5, input5Perm, input5),// max
                              svsel_s32(mask5, input5, input5Perm));// min

            input5_val = svsel_s32(ffttout,
                              svsel_s32(mask5, input5Perm_val, input5_val),// max
                              svsel_s32(mask5, input5_val, input5Perm_val));// min

            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

            input6 = svsel_s32(ffttout,
                              svsel_s32(mask6, input6Perm, input6),// max
                              svsel_s32(mask6, input6, input6Perm));// min

            input6_val = svsel_s32(ffttout,
                              svsel_s32(mask6, input6Perm_val, input6_val),// max
                              svsel_s32(mask6, input6_val, input6Perm_val));// min

            const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t input7Perm_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, input7Perm);

            input7 = svsel_s32(ffttout,
                              svsel_s32(mask7, input7Perm, input7),// max
                              svsel_s32(mask7, input7, input7Perm));// min

            input7_val = svsel_s32(ffttout,
                              svsel_s32(mask7, input7Perm_val, input7_val),// max
                              svsel_s32(mask7, input7_val, input7Perm_val));// min

            const svint32_t input8Perm = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
            const svint32_t input8Perm_val = svtbl_s32(input8_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask8 = svcmplt_s32(svptrue_b32(), input8, input8Perm);

            input8 = svsel_s32(ffttout,
                              svsel_s32(mask8, input8Perm, input8),// max
                              svsel_s32(mask8, input8, input8Perm));// min

            input8_val = svsel_s32(ffttout,
                              svsel_s32(mask8, input8Perm_val, input8_val),// max
                              svsel_s32(mask8, input8_val, input8Perm_val));// min
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b32(ffttout,ffttout);
            vecincout = svsel_s32(ffttout,
                                      svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                                      svadd_n_s32_z(svptrue_b32(), vecincout, stepout));

            {
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(ffttout,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(ffttout,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(ffttout,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(ffttout,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(ffttout,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(ffttout,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min

                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

                input4 = svsel_s32(ffttout,
                                  svsel_s32(mask4, input4Perm, input4),// max
                                  svsel_s32(mask4, input4, input4Perm));// min

                input4_val = svsel_s32(ffttout,
                                  svsel_s32(mask4, input4Perm_val, input4_val),// max
                                  svsel_s32(mask4, input4_val, input4Perm_val));// min

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

                input5 = svsel_s32(ffttout,
                                  svsel_s32(mask5, input5Perm, input5),// max
                                  svsel_s32(mask5, input5, input5Perm));// min

                input5_val = svsel_s32(ffttout,
                                  svsel_s32(mask5, input5Perm_val, input5_val),// max
                                  svsel_s32(mask5, input5_val, input5Perm_val));// min

                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

                input6 = svsel_s32(ffttout,
                                  svsel_s32(mask6, input6Perm, input6),// max
                                  svsel_s32(mask6, input6, input6Perm));// min

                input6_val = svsel_s32(ffttout,
                                  svsel_s32(mask6, input6Perm_val, input6_val),// max
                                  svsel_s32(mask6, input6_val, input6Perm_val));// min

                const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
                const svint32_t input7Perm_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, input7Perm);

                input7 = svsel_s32(ffttout,
                                  svsel_s32(mask7, input7Perm, input7),// max
                                  svsel_s32(mask7, input7, input7Perm));// min

                input7_val = svsel_s32(ffttout,
                                  svsel_s32(mask7, input7Perm_val, input7_val),// max
                                  svsel_s32(mask7, input7_val, input7Perm_val));// min

                const svint32_t input8Perm = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
                const svint32_t input8Perm_val = svtbl_s32(input8_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask8 = svcmplt_s32(svptrue_b32(), input8, input8Perm);

                input8 = svsel_s32(ffttout,
                                  svsel_s32(mask8, input8Perm, input8),// max
                                  svsel_s32(mask8, input8, input8Perm));// min

                input8_val = svsel_s32(ffttout,
                                  svsel_s32(mask8, input8Perm_val, input8_val),// max
                                  svsel_s32(mask8, input8_val, input8Perm_val));// min
            }

            svbool_t fftt = svuzp2_b32(ffttout,ffttout);

            svint32_t vecinc = svdup_s32(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min

                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

                input4 = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm, input4),// max
                                  svsel_s32(mask4, input4, input4Perm));// min

                input4_val = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm_val, input4_val),// max
                                  svsel_s32(mask4, input4_val, input4Perm_val));// min

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

                input5 = svsel_s32(fftt,
                                  svsel_s32(mask5, input5Perm, input5),// max
                                  svsel_s32(mask5, input5, input5Perm));// min

                input5_val = svsel_s32(fftt,
                                  svsel_s32(mask5, input5Perm_val, input5_val),// max
                                  svsel_s32(mask5, input5_val, input5Perm_val));// min

                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

                input6 = svsel_s32(fftt,
                                  svsel_s32(mask6, input6Perm, input6),// max
                                  svsel_s32(mask6, input6, input6Perm));// min

                input6_val = svsel_s32(fftt,
                                  svsel_s32(mask6, input6Perm_val, input6_val),// max
                                  svsel_s32(mask6, input6_val, input6Perm_val));// min

                const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
                const svint32_t input7Perm_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, input7Perm);

                input7 = svsel_s32(fftt,
                                  svsel_s32(mask7, input7Perm, input7),// max
                                  svsel_s32(mask7, input7, input7Perm));// min

                input7_val = svsel_s32(fftt,
                                  svsel_s32(mask7, input7Perm_val, input7_val),// max
                                  svsel_s32(mask7, input7_val, input7Perm_val));// min

                const svint32_t input8Perm = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
                const svint32_t input8Perm_val = svtbl_s32(input8_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask8 = svcmplt_s32(svptrue_b32(), input8, input8Perm);

                input8 = svsel_s32(fftt,
                                  svsel_s32(mask8, input8Perm, input8),// max
                                  svsel_s32(mask8, input8, input8Perm));// min

                input8_val = svsel_s32(fftt,
                                  svsel_s32(mask8, input8Perm_val, input8_val),// max
                                  svsel_s32(mask8, input8_val, input8Perm_val));// min

                fftt = svuzp2_b32(fftt,fftt);
                vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);

                input = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm, input),// max
                                  svsel_s32(mask, input, inputPerm));// min

                input_val = svsel_s32(fftt,
                                  svsel_s32(mask, inputPerm_val, input_val),// max
                                  svsel_s32(mask, input_val, inputPerm_val));// min

                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, input2Perm);

                input2 = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm, input2),// max
                                  svsel_s32(mask2, input2, input2Perm));// min

                input2_val = svsel_s32(fftt,
                                  svsel_s32(mask2, input2Perm_val, input2_val),// max
                                  svsel_s32(mask2, input2_val, input2Perm_val));// min

                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

                input3 = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm, input3),// max
                                  svsel_s32(mask3, input3, input3Perm));// min

                input3_val = svsel_s32(fftt,
                                  svsel_s32(mask3, input3Perm_val, input3_val),// max
                                  svsel_s32(mask3, input3_val, input3Perm_val));// min

                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

                input4 = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm, input4),// max
                                  svsel_s32(mask4, input4, input4Perm));// min

                input4_val = svsel_s32(fftt,
                                  svsel_s32(mask4, input4Perm_val, input4_val),// max
                                  svsel_s32(mask4, input4_val, input4Perm_val));// min

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

                input5 = svsel_s32(fftt,
                                  svsel_s32(mask5, input5Perm, input5),// max
                                  svsel_s32(mask5, input5, input5Perm));// min

                input5_val = svsel_s32(fftt,
                                  svsel_s32(mask5, input5Perm_val, input5_val),// max
                                  svsel_s32(mask5, input5_val, input5Perm_val));// min

                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

                input6 = svsel_s32(fftt,
                                  svsel_s32(mask6, input6Perm, input6),// max
                                  svsel_s32(mask6, input6, input6Perm));// min

                input6_val = svsel_s32(fftt,
                                  svsel_s32(mask6, input6Perm_val, input6_val),// max
                                  svsel_s32(mask6, input6_val, input6Perm_val));// min

                const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
                const svint32_t input7Perm_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, input7Perm);

                input7 = svsel_s32(fftt,
                                  svsel_s32(mask7, input7Perm, input7),// max
                                  svsel_s32(mask7, input7, input7Perm));// min

                input7_val = svsel_s32(fftt,
                                  svsel_s32(mask7, input7Perm_val, input7_val),// max
                                  svsel_s32(mask7, input7_val, input7Perm_val));// min

                const svint32_t input8Perm = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
                const svint32_t input8Perm_val = svtbl_s32(input8_val, svreinterpret_u32_s32(vecpermute));

                svbool_t mask8 = svcmplt_s32(svptrue_b32(), input8, input8Perm);

                input8 = svsel_s32(fftt,
                                  svsel_s32(mask8, input8Perm, input8),// max
                                  svsel_s32(mask8, input8, input8Perm));// min

                input8_val = svsel_s32(fftt,
                                  svsel_s32(mask8, input8Perm_val, input8_val),// max
                                  svsel_s32(mask8, input8_val, input8Perm_val));// min
            }
        }
    }
    {
        CoreExchange(input, input2,
                     input_val, input2_val);
    }
    {
        CoreExchange(input3, input4,
                     input3_val, input4_val);
    }
    {
        CoreExchange(input5, input6,
                     input5_val, input6_val);
    }
    {
        CoreExchange(input7, input8,
                     input7_val, input8_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min

            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm, input3),// max
                              svsel_s32(mask3, input3, input3Perm));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm_val, input3_val),// max
                              svsel_s32(mask3, input3_val, input3Perm_val));// min

            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm, input4),// max
                              svsel_s32(mask4, input4, input4Perm));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm_val, input4_val),// max
                              svsel_s32(mask4, input4_val, input4Perm_val));// min

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm, input5),// max
                              svsel_s32(mask5, input5, input5Perm));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm_val, input5_val),// max
                              svsel_s32(mask5, input5_val, input5Perm_val));// min

            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm, input6),// max
                              svsel_s32(mask6, input6, input6Perm));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm_val, input6_val),// max
                              svsel_s32(mask6, input6_val, input6Perm_val));// min

            const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t input7Perm_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, input7Perm);

            input7 = svsel_s32(fftt,
                              svsel_s32(mask7, input7Perm, input7),// max
                              svsel_s32(mask7, input7, input7Perm));// min

            input7_val = svsel_s32(fftt,
                              svsel_s32(mask7, input7Perm_val, input7_val),// max
                              svsel_s32(mask7, input7_val, input7Perm_val));// min

            const svint32_t input8Perm = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
            const svint32_t input8Perm_val = svtbl_s32(input8_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask8 = svcmplt_s32(svptrue_b32(), input8, input8Perm);

            input8 = svsel_s32(fftt,
                              svsel_s32(mask8, input8Perm, input8),// max
                              svsel_s32(mask8, input8, input8Perm));// min

            input8_val = svsel_s32(fftt,
                              svsel_s32(mask8, input8Perm_val, input8_val),// max
                              svsel_s32(mask8, input8_val, input8Perm_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min

            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, input3Perm);

            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm, input3),// max
                              svsel_s32(mask3, input3, input3Perm));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, input3Perm_val, input3_val),// max
                              svsel_s32(mask3, input3_val, input3Perm_val));// min

            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, input4Perm);

            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm, input4),// max
                              svsel_s32(mask4, input4, input4Perm));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, input4Perm_val, input4_val),// max
                              svsel_s32(mask4, input4_val, input4Perm_val));// min

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm, input5),// max
                              svsel_s32(mask5, input5, input5Perm));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm_val, input5_val),// max
                              svsel_s32(mask5, input5_val, input5Perm_val));// min

            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm, input6),// max
                              svsel_s32(mask6, input6, input6Perm));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm_val, input6_val),// max
                              svsel_s32(mask6, input6_val, input6Perm_val));// min

            const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t input7Perm_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, input7Perm);

            input7 = svsel_s32(fftt,
                              svsel_s32(mask7, input7Perm, input7),// max
                              svsel_s32(mask7, input7, input7Perm));// min

            input7_val = svsel_s32(fftt,
                              svsel_s32(mask7, input7Perm_val, input7_val),// max
                              svsel_s32(mask7, input7_val, input7Perm_val));// min

            const svint32_t input8Perm = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
            const svint32_t input8Perm_val = svtbl_s32(input8_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask8 = svcmplt_s32(svptrue_b32(), input8, input8Perm);

            input8 = svsel_s32(fftt,
                              svsel_s32(mask8, input8Perm, input8),// max
                              svsel_s32(mask8, input8, input8Perm));// min

            input8_val = svsel_s32(fftt,
                              svsel_s32(mask8, input8Perm_val, input8_val),// max
                              svsel_s32(mask8, input8_val, input8Perm_val));// min
        }
    }
    {
        CoreExchange(input, input4,
                     input_val, input4_val);
        CoreExchange(input2, input3,
                     input2_val, input3_val);
    }
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
    }
    {
        CoreExchangeEnd(input3, input4,
                        input3_val, input4_val);
    }
    {
        CoreExchange(input5, input8,
                     input5_val, input8_val);
        CoreExchange(input6, input7,
                     input6_val, input7_val);
    }
    {
        CoreExchangeEnd(input5, input6,
                        input5_val, input6_val);
    }
    {
        CoreExchangeEnd(input7, input8,
                        input7_val, input8_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm, input5),// max
                              svsel_s32(mask5, input5, input5Perm));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm_val, input5_val),// max
                              svsel_s32(mask5, input5_val, input5Perm_val));// min

            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm, input6),// max
                              svsel_s32(mask6, input6, input6Perm));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm_val, input6_val),// max
                              svsel_s32(mask6, input6_val, input6Perm_val));// min

            const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t input7Perm_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, input7Perm);

            input7 = svsel_s32(fftt,
                              svsel_s32(mask7, input7Perm, input7),// max
                              svsel_s32(mask7, input7, input7Perm));// min

            input7_val = svsel_s32(fftt,
                              svsel_s32(mask7, input7Perm_val, input7_val),// max
                              svsel_s32(mask7, input7_val, input7Perm_val));// min

            const svint32_t input8Perm = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
            const svint32_t input8Perm_val = svtbl_s32(input8_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask8 = svcmplt_s32(svptrue_b32(), input8, input8Perm);

            input8 = svsel_s32(fftt,
                              svsel_s32(mask8, input8Perm, input8),// max
                              svsel_s32(mask8, input8, input8Perm));// min

            input8_val = svsel_s32(fftt,
                              svsel_s32(mask8, input8Perm_val, input8_val),// max
                              svsel_s32(mask8, input8_val, input8Perm_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input5Perm_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, input5Perm);

            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm, input5),// max
                              svsel_s32(mask5, input5, input5Perm));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, input5Perm_val, input5_val),// max
                              svsel_s32(mask5, input5_val, input5Perm_val));// min

            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, input6Perm);

            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm, input6),// max
                              svsel_s32(mask6, input6, input6Perm));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, input6Perm_val, input6_val),// max
                              svsel_s32(mask6, input6_val, input6Perm_val));// min

            const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t input7Perm_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, input7Perm);

            input7 = svsel_s32(fftt,
                              svsel_s32(mask7, input7Perm, input7),// max
                              svsel_s32(mask7, input7, input7Perm));// min

            input7_val = svsel_s32(fftt,
                              svsel_s32(mask7, input7Perm_val, input7_val),// max
                              svsel_s32(mask7, input7_val, input7Perm_val));// min

            const svint32_t input8Perm = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
            const svint32_t input8Perm_val = svtbl_s32(input8_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask8 = svcmplt_s32(svptrue_b32(), input8, input8Perm);

            input8 = svsel_s32(fftt,
                              svsel_s32(mask8, input8Perm, input8),// max
                              svsel_s32(mask8, input8, input8Perm));// min

            input8_val = svsel_s32(fftt,
                              svsel_s32(mask8, input8Perm_val, input8_val),// max
                              svsel_s32(mask8, input8_val, input8Perm_val));// min
        }
    }
#endif
    {
        CoreExchange(input, input8,
                     input_val, input8_val);
        CoreExchange(input2, input7,
                     input2_val, input7_val);
        CoreExchange(input3, input6,
                     input3_val, input6_val);
        CoreExchange(input4, input5,
                     input4_val, input5_val);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4,
                  input_val, input2_val, input3_val, input4_val);
    CoreSmallEnd4(input5, input6, input7, input8,
                  input5_val, input6_val, input7_val, input8_val);
#else
    {
        CoreExchangeEnd(input, input3,
                        input_val, input3_val);
        CoreExchangeEnd(input2, input4,
                        input2_val, input4_val);
    }
    {
        CoreExchangeEnd(input, input2,
                        input_val, input2_val);
        CoreExchangeEnd(input3, input4,
                        input3_val, input4_val);
    }
    {
        CoreExchangeEnd(input5, input7,
                        input5_val, input7_val);
        CoreExchangeEnd(input6, input8,
                        input6_val, input8_val);
    }
    {
        CoreExchangeEnd(input5, input6,
                        input5_val, input6_val);
        CoreExchangeEnd(input7, input8,
                        input7_val, input8_val);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = SortSVE::getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7 = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm8 = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm8_val = svtbl_s32(input8_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);
            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, inputPerm5);
            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, inputPerm6);
            svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, inputPerm7);
            svbool_t mask8 = svcmplt_s32(svptrue_b32(), input8, inputPerm8);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5, input5),// max
                              svsel_s32(mask5, input5, inputPerm5));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5_val, input5_val),// max
                              svsel_s32(mask5, input5_val, inputPerm5_val));// min
            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6, input6),// max
                              svsel_s32(mask6, input6, inputPerm6));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6_val, input6_val),// max
                              svsel_s32(mask6, input6_val, inputPerm6_val));// min
            input7 = svsel_s32(fftt,
                              svsel_s32(mask7, inputPerm7, input7),// max
                              svsel_s32(mask7, input7, inputPerm7));// min

            input7_val = svsel_s32(fftt,
                              svsel_s32(mask7, inputPerm7_val, input7_val),// max
                              svsel_s32(mask7, input7_val, inputPerm7_val));// min
            input8 = svsel_s32(fftt,
                              svsel_s32(mask8, inputPerm8, input8),// max
                              svsel_s32(mask8, input8, inputPerm8));// min

            input8_val = svsel_s32(fftt,
                              svsel_s32(mask8, inputPerm8_val, input8_val),// max
                              svsel_s32(mask8, input8_val, inputPerm8_val));// min

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm_val = svtbl_s32(input_val, svreinterpret_u32_s32(vecpermute));

            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2_val = svtbl_s32(input2_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3_val = svtbl_s32(input3_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4_val = svtbl_s32(input4_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5_val = svtbl_s32(input5_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6_val = svtbl_s32(input6_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7 = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7_val = svtbl_s32(input7_val, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm8 = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm8_val = svtbl_s32(input8_val, svreinterpret_u32_s32(vecpermute));

            svbool_t mask = svcmplt_s32(svptrue_b32(), input, inputPerm);
            svbool_t mask2 = svcmplt_s32(svptrue_b32(), input2, inputPerm2);
            svbool_t mask3 = svcmplt_s32(svptrue_b32(), input3, inputPerm3);
            svbool_t mask4 = svcmplt_s32(svptrue_b32(), input4, inputPerm4);
            svbool_t mask5 = svcmplt_s32(svptrue_b32(), input5, inputPerm5);
            svbool_t mask6 = svcmplt_s32(svptrue_b32(), input6, inputPerm6);
            svbool_t mask7 = svcmplt_s32(svptrue_b32(), input7, inputPerm7);
            svbool_t mask8 = svcmplt_s32(svptrue_b32(), input8, inputPerm8);

            input = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm, input),// max
                              svsel_s32(mask, input, inputPerm));// min

            input_val = svsel_s32(fftt,
                              svsel_s32(mask, inputPerm_val, input_val),// max
                              svsel_s32(mask, input_val, inputPerm_val));// min


            input2 = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2, input2),// max
                              svsel_s32(mask2, input2, inputPerm2));// min

            input2_val = svsel_s32(fftt,
                              svsel_s32(mask2, inputPerm2_val, input2_val),// max
                              svsel_s32(mask2, input2_val, inputPerm2_val));// min
            input3 = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3, input3),// max
                              svsel_s32(mask3, input3, inputPerm3));// min

            input3_val = svsel_s32(fftt,
                              svsel_s32(mask3, inputPerm3_val, input3_val),// max
                              svsel_s32(mask3, input3_val, inputPerm3_val));// min
            input4 = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4, input4),// max
                              svsel_s32(mask4, input4, inputPerm4));// min

            input4_val = svsel_s32(fftt,
                              svsel_s32(mask4, inputPerm4_val, input4_val),// max
                              svsel_s32(mask4, input4_val, inputPerm4_val));// min
            input5 = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5, input5),// max
                              svsel_s32(mask5, input5, inputPerm5));// min

            input5_val = svsel_s32(fftt,
                              svsel_s32(mask5, inputPerm5_val, input5_val),// max
                              svsel_s32(mask5, input5_val, inputPerm5_val));// min
            input6 = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6, input6),// max
                              svsel_s32(mask6, input6, inputPerm6));// min

            input6_val = svsel_s32(fftt,
                              svsel_s32(mask6, inputPerm6_val, input6_val),// max
                              svsel_s32(mask6, input6_val, inputPerm6_val));// min
            input7 = svsel_s32(fftt,
                              svsel_s32(mask7, inputPerm7, input7),// max
                              svsel_s32(mask7, input7, inputPerm7));// min

            input7_val = svsel_s32(fftt,
                              svsel_s32(mask7, inputPerm7_val, input7_val),// max
                              svsel_s32(mask7, input7_val, inputPerm7_val));// min
            input8 = svsel_s32(fftt,
                              svsel_s32(mask8, inputPerm8, input8),// max
                              svsel_s32(mask8, input8, inputPerm8));// min

            input8_val = svsel_s32(fftt,
                              svsel_s32(mask8, inputPerm8_val, input8_val),// max
                              svsel_s32(mask8, input8_val, inputPerm8_val));// min
        }
    }
#endif
}

inline void CoreSmallSort9(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                           svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                           svint32_t& input9,
                           svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                           svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val,
                           svint32_t& input9_val){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8,
                   input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSort(input9,
                           input9_val);
    {
        CoreExchange(input8, input9,
                     input8_val, input9_val);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8,
                  input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSortEnd(input9,
                  input9_val);
}

inline void CoreSmallSort10(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10,
                            svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                            svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val,
                            svint32_t& input9_val, svint32_t& input10_val){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8,
                   input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSort2(input9, input10,
                   input9_val, input10_val);
    {
        CoreExchange(input7, input10,
                     input7_val, input10_val);
        CoreExchange(input8, input9,
                     input8_val, input9_val);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8,
                  input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallEnd2(input9, input10,
                  input9_val, input10_val);
}

inline void CoreSmallSort11(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11,
                            svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                            svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val,
                            svint32_t& input9_val, svint32_t& input10_val, svint32_t& input11_val){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8,
                   input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSort3(input9, input10, input11,
                   input9_val, input10_val, input11_val);
    {
        CoreExchange(input6, input11,
                     input6_val, input11_val);
        CoreExchange(input7, input10,
                     input7_val, input10_val);
        CoreExchange(input8, input9,
                     input8_val, input9_val);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8,
                  input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallEnd3(input9, input10, input11,
                  input9_val, input10_val, input11_val);
}

inline void CoreSmallSort12(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                            svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                            svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val,
                            svint32_t& input9_val, svint32_t& input10_val, svint32_t& input11_val, svint32_t& input12_val){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8,
                   input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSort4(input9, input10, input11, input12,
                   input9_val, input10_val, input11_val, input12_val);
    {
        CoreExchange(input5, input12,
                     input5_val, input12_val);
        CoreExchange(input6, input11,
                     input6_val, input11_val);
        CoreExchange(input7, input10,
                     input7_val, input10_val);
        CoreExchange(input8, input9,
                     input8_val, input9_val);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8,
                  input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallEnd4(input9, input10, input11, input12,
                  input9_val, input10_val, input11_val, input12_val);
}

inline void CoreSmallSort13(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                            svint32_t& input13,
                            svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                            svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val,
                            svint32_t& input9_val, svint32_t& input10_val, svint32_t& input11_val, svint32_t& input12_val,
                            svint32_t& input13_val){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8,
                   input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSort5(input9, input10, input11, input12, input13,
                   input9_val, input10_val, input11_val, input12_val, input13_val);
    {
        CoreExchange(input4, input13,
                     input4_val, input13_val);
        CoreExchange(input5, input12,
                     input5_val, input12_val);
        CoreExchange(input6, input11,
                     input6_val, input11_val);
        CoreExchange(input7, input10,
                     input7_val, input10_val);
        CoreExchange(input8, input9,
                     input8_val, input9_val);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8,
                  input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallEnd5(input9, input10, input11, input12, input13,
                  input9_val, input10_val, input11_val, input12_val, input13_val);
}

inline void CoreSmallSort14(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                            svint32_t& input13, svint32_t& input14,
                            svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                            svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val,
                            svint32_t& input9_val, svint32_t& input10_val, svint32_t& input11_val, svint32_t& input12_val,
                            svint32_t& input13_val, svint32_t& input14_val){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8,
                   input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSort6(input9, input10, input11, input12, input13, input14,
                   input9_val, input10_val, input11_val, input12_val, input13_val, input14_val);
    {
        CoreExchange(input3, input14,
                     input3_val, input14_val);
        CoreExchange(input4, input13,
                     input4_val, input13_val);
        CoreExchange(input5, input12,
                     input5_val, input12_val);
        CoreExchange(input6, input11,
                     input6_val, input11_val);
        CoreExchange(input7, input10,
                     input7_val, input10_val);
        CoreExchange(input8, input9,
                     input8_val, input9_val);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8,
                  input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallEnd6(input9, input10, input11, input12, input13, input14,
                  input9_val, input10_val, input11_val, input12_val, input13_val, input14_val);
}

inline void CoreSmallSort15(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                            svint32_t& input13, svint32_t& input14, svint32_t& input15,
                            svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                            svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val,
                            svint32_t& input9_val, svint32_t& input10_val, svint32_t& input11_val, svint32_t& input12_val,
                            svint32_t& input13_val, svint32_t& input14_val, svint32_t& input15_val){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8,
                   input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSort7(input9, input10, input11, input12, input13, input14, input15,
                   input9_val, input10_val, input11_val, input12_val, input13_val, input14_val, input15_val);
    {
        CoreExchange(input2, input15,
                     input2_val, input15_val);
        CoreExchange(input3, input14,
                     input3_val, input14_val);
        CoreExchange(input4, input13,
                     input4_val, input13_val);
        CoreExchange(input5, input12,
                     input5_val, input12_val);
        CoreExchange(input6, input11,
                     input6_val, input11_val);
        CoreExchange(input7, input10,
                     input7_val, input10_val);
        CoreExchange(input8, input9,
                     input8_val, input9_val);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8,
                  input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallEnd7(input9, input10, input11, input12, input13, input14, input15,
                  input9_val, input10_val, input11_val, input12_val, input13_val, input14_val, input15_val);
}

inline void CoreSmallSort16(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                            svint32_t& input13, svint32_t& input14, svint32_t& input15, svint32_t& input16,
                            svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                            svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val,
                            svint32_t& input9_val, svint32_t& input10_val, svint32_t& input11_val, svint32_t& input12_val,
                            svint32_t& input13_val, svint32_t& input14_val, svint32_t& input15_val, svint32_t& input16_val){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8,
                   input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSort8(input9, input10, input11, input12, input13, input14, input15, input16,
                   input9_val, input10_val, input11_val, input12_val, input13_val, input14_val, input15_val, input16_val);
    {
        CoreExchange(input, input16,
                     input_val, input16_val);
        CoreExchange(input2, input15,
                     input2_val, input15_val);
        CoreExchange(input3, input14,
                     input3_val, input14_val);
        CoreExchange(input4, input13,
                     input4_val, input13_val);
        CoreExchange(input5, input12,
                     input5_val, input12_val);
        CoreExchange(input6, input11,
                     input6_val, input11_val);
        CoreExchange(input7, input10,
                     input7_val, input10_val);
        CoreExchange(input8, input9,
                     input8_val, input9_val);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8,
                  input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallEnd8(input9, input10, input11, input12, input13, input14, input15, input16,
                  input9_val, input10_val, input11_val, input12_val, input13_val, input14_val, input15_val, input16_val);
}


////////////////////////////////////////////////////////////

inline void CoreSmallSort( int* __restrict__ ptr1, int* __restrict__ values){
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values);
    CoreSmallSort(input0,input0_val);
    svst1_s32(svptrue_b32(),ptr1, input0);
    svst1_s32(svptrue_b32(),values, input0_val);
}


inline void CoreSmallSort2( int* __restrict__ ptr1, int* __restrict__ values){
    const int nbValuesInVec = svcntw();
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+nbValuesInVec);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
    CoreSmallSort2(input0,input1,input0_val,input1_val);
    svst1_s32(svptrue_b32(),ptr1, input0);
    svst1_s32(svptrue_b32(),ptr1+nbValuesInVec,input1);
    svst1_s32(svptrue_b32(),values, input0_val);
    svst1_s32(svptrue_b32(),values+nbValuesInVec,input1_val);
}


inline void CoreSmallSort3( int* __restrict__ ptr1, int* __restrict__ values){
    const int nbValuesInVec = svcntw();
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+nbValuesInVec);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*nbValuesInVec);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
    CoreSmallSort3(input0,input1,input2,input0_val,input1_val,input2_val);
    svst1_s32(svptrue_b32(),ptr1, input0);
    svst1_s32(svptrue_b32(),ptr1+nbValuesInVec,input1);
    svst1_s32(svptrue_b32(),ptr1+2*nbValuesInVec,input2);
    svst1_s32(svptrue_b32(),values, input0_val);
    svst1_s32(svptrue_b32(),values+nbValuesInVec,input1_val);
    svst1_s32(svptrue_b32(),values+2*nbValuesInVec,input2_val);
}


inline void CoreSmallSort4( int* __restrict__ ptr1, int* __restrict__ values){
    const int nbValuesInVec = svcntw();
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+nbValuesInVec);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*nbValuesInVec);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*nbValuesInVec);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
    CoreSmallSort4(input0,input1,input2,input3,input0_val,input1_val,input2_val,input3_val);
    svst1_s32(svptrue_b32(),ptr1, input0);
    svst1_s32(svptrue_b32(),ptr1+nbValuesInVec,input1);
    svst1_s32(svptrue_b32(),ptr1+2*nbValuesInVec,input2);
    svst1_s32(svptrue_b32(),ptr1+3*nbValuesInVec,input3);
    svst1_s32(svptrue_b32(),values, input0_val);
    svst1_s32(svptrue_b32(),values+nbValuesInVec,input1_val);
    svst1_s32(svptrue_b32(),values+2*nbValuesInVec,input2_val);
    svst1_s32(svptrue_b32(),values+3*nbValuesInVec,input3_val);
}


inline void CoreSmallSort5( int* __restrict__ ptr1, int* __restrict__ values){
    const int nbValuesInVec = svcntw();
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+nbValuesInVec);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*nbValuesInVec);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*nbValuesInVec);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*nbValuesInVec);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
    CoreSmallSort5(input0,input1,input2,input3,
                   input4,input0_val,input1_val,input2_val,input3_val,
                   input4_val);
    svst1_s32(svptrue_b32(),ptr1, input0);
    svst1_s32(svptrue_b32(),ptr1+nbValuesInVec,input1);
    svst1_s32(svptrue_b32(),ptr1+2*nbValuesInVec,input2);
    svst1_s32(svptrue_b32(),ptr1+3*nbValuesInVec,input3);
    svst1_s32(svptrue_b32(),ptr1+4*nbValuesInVec,input4);
    svst1_s32(svptrue_b32(),values, input0_val);
    svst1_s32(svptrue_b32(),values+nbValuesInVec,input1_val);
    svst1_s32(svptrue_b32(),values+2*nbValuesInVec,input2_val);
    svst1_s32(svptrue_b32(),values+3*nbValuesInVec,input3_val);
    svst1_s32(svptrue_b32(),values+4*nbValuesInVec,input4_val);
}


inline void CoreSmallSort6( int* __restrict__ ptr1, int* __restrict__ values){
    const int nbValuesInVec = svcntw();
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+nbValuesInVec);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*nbValuesInVec);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*nbValuesInVec);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*nbValuesInVec);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*nbValuesInVec);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
    CoreSmallSort6(input0,input1,input2,input3,
                   input4,input5,input0_val,input1_val,input2_val,input3_val,
                   input4_val,input5_val);
    svst1_s32(svptrue_b32(),ptr1, input0);
    svst1_s32(svptrue_b32(),ptr1+nbValuesInVec,input1);
    svst1_s32(svptrue_b32(),ptr1+2*nbValuesInVec,input2);
    svst1_s32(svptrue_b32(),ptr1+3*nbValuesInVec,input3);
    svst1_s32(svptrue_b32(),ptr1+4*nbValuesInVec,input4);
    svst1_s32(svptrue_b32(),ptr1+5*nbValuesInVec,input5);
    svst1_s32(svptrue_b32(),values, input0_val);
    svst1_s32(svptrue_b32(),values+nbValuesInVec,input1_val);
    svst1_s32(svptrue_b32(),values+2*nbValuesInVec,input2_val);
    svst1_s32(svptrue_b32(),values+3*nbValuesInVec,input3_val);
    svst1_s32(svptrue_b32(),values+4*nbValuesInVec,input4_val);
    svst1_s32(svptrue_b32(),values+5*nbValuesInVec,input5_val);
}


inline void CoreSmallSort7( int* __restrict__ ptr1, int* __restrict__ values){
    const int nbValuesInVec = svcntw();
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+nbValuesInVec);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*nbValuesInVec);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*nbValuesInVec);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*nbValuesInVec);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*nbValuesInVec);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*nbValuesInVec);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
    CoreSmallSort7(input0,input1,input2,input3,
                   input4,input5,input6,input0_val,input1_val,input2_val,input3_val,
                   input4_val,input5_val,input6_val);
    svst1_s32(svptrue_b32(),ptr1, input0);
    svst1_s32(svptrue_b32(),ptr1+nbValuesInVec,input1);
    svst1_s32(svptrue_b32(),ptr1+2*nbValuesInVec,input2);
    svst1_s32(svptrue_b32(),ptr1+3*nbValuesInVec,input3);
    svst1_s32(svptrue_b32(),ptr1+4*nbValuesInVec,input4);
    svst1_s32(svptrue_b32(),ptr1+5*nbValuesInVec,input5);
    svst1_s32(svptrue_b32(),ptr1+6*nbValuesInVec,input6);
    svst1_s32(svptrue_b32(),values, input0_val);
    svst1_s32(svptrue_b32(),values+nbValuesInVec,input1_val);
    svst1_s32(svptrue_b32(),values+2*nbValuesInVec,input2_val);
    svst1_s32(svptrue_b32(),values+3*nbValuesInVec,input3_val);
    svst1_s32(svptrue_b32(),values+4*nbValuesInVec,input4_val);
    svst1_s32(svptrue_b32(),values+5*nbValuesInVec,input5_val);
    svst1_s32(svptrue_b32(),values+6*nbValuesInVec,input6_val);
}


inline void CoreSmallSort8( int* __restrict__ ptr1, int* __restrict__ values){
    const int nbValuesInVec = svcntw();
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+nbValuesInVec);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*nbValuesInVec);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*nbValuesInVec);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*nbValuesInVec);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*nbValuesInVec);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*nbValuesInVec);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*nbValuesInVec);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
    CoreSmallSort8(input0,input1,input2,input3,
                   input4,input5,input6,input7,input0_val,input1_val,input2_val,input3_val,
                   input4_val,input5_val,input6_val,input7_val);
    svst1_s32(svptrue_b32(),ptr1, input0);
    svst1_s32(svptrue_b32(),ptr1+nbValuesInVec,input1);
    svst1_s32(svptrue_b32(),ptr1+2*nbValuesInVec,input2);
    svst1_s32(svptrue_b32(),ptr1+3*nbValuesInVec,input3);
    svst1_s32(svptrue_b32(),ptr1+4*nbValuesInVec,input4);
    svst1_s32(svptrue_b32(),ptr1+5*nbValuesInVec,input5);
    svst1_s32(svptrue_b32(),ptr1+6*nbValuesInVec,input6);
    svst1_s32(svptrue_b32(),ptr1+7*nbValuesInVec,input7);
    svst1_s32(svptrue_b32(),values, input0_val);
    svst1_s32(svptrue_b32(),values+nbValuesInVec,input1_val);
    svst1_s32(svptrue_b32(),values+2*nbValuesInVec,input2_val);
    svst1_s32(svptrue_b32(),values+3*nbValuesInVec,input3_val);
    svst1_s32(svptrue_b32(),values+4*nbValuesInVec,input4_val);
    svst1_s32(svptrue_b32(),values+5*nbValuesInVec,input5_val);
    svst1_s32(svptrue_b32(),values+6*nbValuesInVec,input6_val);
    svst1_s32(svptrue_b32(),values+7*nbValuesInVec,input7_val);
}


inline void CoreSmallSort9( int* __restrict__ ptr1, int* __restrict__ values){
    const int nbValuesInVec = svcntw();
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+nbValuesInVec);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*nbValuesInVec);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*nbValuesInVec);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*nbValuesInVec);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*nbValuesInVec);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*nbValuesInVec);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*nbValuesInVec);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr1+8*nbValuesInVec);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
    svint32_t input8_val = svld1_s32(svptrue_b32(),values+8*nbValuesInVec);
    CoreSmallSort9(input0,input1,input2,input3,
                   input4,input5,input6,input7,
                   input8,input0_val,input1_val,input2_val,input3_val,
                   input4_val,input5_val,input6_val,input7_val,
                   input8_val);
    svst1_s32(svptrue_b32(),ptr1, input0);
    svst1_s32(svptrue_b32(),ptr1+nbValuesInVec,input1);
    svst1_s32(svptrue_b32(),ptr1+2*nbValuesInVec,input2);
    svst1_s32(svptrue_b32(),ptr1+3*nbValuesInVec,input3);
    svst1_s32(svptrue_b32(),ptr1+4*nbValuesInVec,input4);
    svst1_s32(svptrue_b32(),ptr1+5*nbValuesInVec,input5);
    svst1_s32(svptrue_b32(),ptr1+6*nbValuesInVec,input6);
    svst1_s32(svptrue_b32(),ptr1+7*nbValuesInVec,input7);
    svst1_s32(svptrue_b32(),ptr1+8*nbValuesInVec,input8);
    svst1_s32(svptrue_b32(),values, input0_val);
    svst1_s32(svptrue_b32(),values+nbValuesInVec,input1_val);
    svst1_s32(svptrue_b32(),values+2*nbValuesInVec,input2_val);
    svst1_s32(svptrue_b32(),values+3*nbValuesInVec,input3_val);
    svst1_s32(svptrue_b32(),values+4*nbValuesInVec,input4_val);
    svst1_s32(svptrue_b32(),values+5*nbValuesInVec,input5_val);
    svst1_s32(svptrue_b32(),values+6*nbValuesInVec,input6_val);
    svst1_s32(svptrue_b32(),values+7*nbValuesInVec,input7_val);
    svst1_s32(svptrue_b32(),values+8*nbValuesInVec,input8_val);
}


inline void CoreSmallSort10( int* __restrict__ ptr1, int* __restrict__ values){
    const int nbValuesInVec = svcntw();
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+nbValuesInVec);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*nbValuesInVec);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*nbValuesInVec);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*nbValuesInVec);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*nbValuesInVec);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*nbValuesInVec);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*nbValuesInVec);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr1+8*nbValuesInVec);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr1+9*nbValuesInVec);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
    svint32_t input8_val = svld1_s32(svptrue_b32(),values+8*nbValuesInVec);
    svint32_t input9_val = svld1_s32(svptrue_b32(),values+9*nbValuesInVec);
    CoreSmallSort10(input0,input1,input2,input3,
                    input4,input5,input6,input7,
                    input8,input9,input0_val,input1_val,input2_val,input3_val,
                    input4_val,input5_val,input6_val,input7_val,
                    input8_val,input9_val);
    svst1_s32(svptrue_b32(),ptr1, input0);
    svst1_s32(svptrue_b32(),ptr1+nbValuesInVec,input1);
    svst1_s32(svptrue_b32(),ptr1+2*nbValuesInVec,input2);
    svst1_s32(svptrue_b32(),ptr1+3*nbValuesInVec,input3);
    svst1_s32(svptrue_b32(),ptr1+4*nbValuesInVec,input4);
    svst1_s32(svptrue_b32(),ptr1+5*nbValuesInVec,input5);
    svst1_s32(svptrue_b32(),ptr1+6*nbValuesInVec,input6);
    svst1_s32(svptrue_b32(),ptr1+7*nbValuesInVec,input7);
    svst1_s32(svptrue_b32(),ptr1+8*nbValuesInVec,input8);
    svst1_s32(svptrue_b32(),ptr1+9*nbValuesInVec,input9);
    svst1_s32(svptrue_b32(),values, input0_val);
    svst1_s32(svptrue_b32(),values+nbValuesInVec,input1_val);
    svst1_s32(svptrue_b32(),values+2*nbValuesInVec,input2_val);
    svst1_s32(svptrue_b32(),values+3*nbValuesInVec,input3_val);
    svst1_s32(svptrue_b32(),values+4*nbValuesInVec,input4_val);
    svst1_s32(svptrue_b32(),values+5*nbValuesInVec,input5_val);
    svst1_s32(svptrue_b32(),values+6*nbValuesInVec,input6_val);
    svst1_s32(svptrue_b32(),values+7*nbValuesInVec,input7_val);
    svst1_s32(svptrue_b32(),values+8*nbValuesInVec,input8_val);
    svst1_s32(svptrue_b32(),values+9*nbValuesInVec,input9_val);
}


inline void CoreSmallSort11( int* __restrict__ ptr1, int* __restrict__ values){
    const int nbValuesInVec = svcntw();
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+nbValuesInVec);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*nbValuesInVec);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*nbValuesInVec);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*nbValuesInVec);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*nbValuesInVec);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*nbValuesInVec);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*nbValuesInVec);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr1+8*nbValuesInVec);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr1+9*nbValuesInVec);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr1+10*nbValuesInVec);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
    svint32_t input8_val = svld1_s32(svptrue_b32(),values+8*nbValuesInVec);
    svint32_t input9_val = svld1_s32(svptrue_b32(),values+9*nbValuesInVec);
    svint32_t input10_val = svld1_s32(svptrue_b32(),values+10*nbValuesInVec);
    CoreSmallSort11(input0,input1,input2,input3,
                    input4,input5,input6,input7,
                    input8,input9,input10,input0_val,input1_val,input2_val,input3_val,
                    input4_val,input5_val,input6_val,input7_val,
                    input8_val,input9_val,input10_val);
    svst1_s32(svptrue_b32(),ptr1, input0);
    svst1_s32(svptrue_b32(),ptr1+nbValuesInVec,input1);
    svst1_s32(svptrue_b32(),ptr1+2*nbValuesInVec,input2);
    svst1_s32(svptrue_b32(),ptr1+3*nbValuesInVec,input3);
    svst1_s32(svptrue_b32(),ptr1+4*nbValuesInVec,input4);
    svst1_s32(svptrue_b32(),ptr1+5*nbValuesInVec,input5);
    svst1_s32(svptrue_b32(),ptr1+6*nbValuesInVec,input6);
    svst1_s32(svptrue_b32(),ptr1+7*nbValuesInVec,input7);
    svst1_s32(svptrue_b32(),ptr1+8*nbValuesInVec,input8);
    svst1_s32(svptrue_b32(),ptr1+9*nbValuesInVec,input9);
    svst1_s32(svptrue_b32(),ptr1+10*nbValuesInVec,input10);
    svst1_s32(svptrue_b32(),values, input0_val);
    svst1_s32(svptrue_b32(),values+nbValuesInVec,input1_val);
    svst1_s32(svptrue_b32(),values+2*nbValuesInVec,input2_val);
    svst1_s32(svptrue_b32(),values+3*nbValuesInVec,input3_val);
    svst1_s32(svptrue_b32(),values+4*nbValuesInVec,input4_val);
    svst1_s32(svptrue_b32(),values+5*nbValuesInVec,input5_val);
    svst1_s32(svptrue_b32(),values+6*nbValuesInVec,input6_val);
    svst1_s32(svptrue_b32(),values+7*nbValuesInVec,input7_val);
    svst1_s32(svptrue_b32(),values+8*nbValuesInVec,input8_val);
    svst1_s32(svptrue_b32(),values+9*nbValuesInVec,input9_val);
    svst1_s32(svptrue_b32(),values+10*nbValuesInVec,input10_val);
}


inline void CoreSmallSort12( int* __restrict__ ptr1, int* __restrict__ values){
    const int nbValuesInVec = svcntw();
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+nbValuesInVec);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*nbValuesInVec);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*nbValuesInVec);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*nbValuesInVec);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*nbValuesInVec);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*nbValuesInVec);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*nbValuesInVec);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr1+8*nbValuesInVec);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr1+9*nbValuesInVec);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr1+10*nbValuesInVec);
    svint32_t input11 = svld1_s32(svptrue_b32(),ptr1+11*nbValuesInVec);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
    svint32_t input8_val = svld1_s32(svptrue_b32(),values+8*nbValuesInVec);
    svint32_t input9_val = svld1_s32(svptrue_b32(),values+9*nbValuesInVec);
    svint32_t input10_val = svld1_s32(svptrue_b32(),values+10*nbValuesInVec);
    svint32_t input11_val = svld1_s32(svptrue_b32(),values+11*nbValuesInVec);
    CoreSmallSort12(input0,input1,input2,input3,
                    input4,input5,input6,input7,
                    input8,input9,input10,input11,input0_val,input1_val,input2_val,input3_val,
                    input4_val,input5_val,input6_val,input7_val,
                    input8_val,input9_val,input10_val,input11_val);
    svst1_s32(svptrue_b32(),ptr1, input0);
    svst1_s32(svptrue_b32(),ptr1+nbValuesInVec,input1);
    svst1_s32(svptrue_b32(),ptr1+2*nbValuesInVec,input2);
    svst1_s32(svptrue_b32(),ptr1+3*nbValuesInVec,input3);
    svst1_s32(svptrue_b32(),ptr1+4*nbValuesInVec,input4);
    svst1_s32(svptrue_b32(),ptr1+5*nbValuesInVec,input5);
    svst1_s32(svptrue_b32(),ptr1+6*nbValuesInVec,input6);
    svst1_s32(svptrue_b32(),ptr1+7*nbValuesInVec,input7);
    svst1_s32(svptrue_b32(),ptr1+8*nbValuesInVec,input8);
    svst1_s32(svptrue_b32(),ptr1+9*nbValuesInVec,input9);
    svst1_s32(svptrue_b32(),ptr1+10*nbValuesInVec,input10);
    svst1_s32(svptrue_b32(),ptr1+11*nbValuesInVec,input11);
    svst1_s32(svptrue_b32(),values, input0_val);
    svst1_s32(svptrue_b32(),values+nbValuesInVec,input1_val);
    svst1_s32(svptrue_b32(),values+2*nbValuesInVec,input2_val);
    svst1_s32(svptrue_b32(),values+3*nbValuesInVec,input3_val);
    svst1_s32(svptrue_b32(),values+4*nbValuesInVec,input4_val);
    svst1_s32(svptrue_b32(),values+5*nbValuesInVec,input5_val);
    svst1_s32(svptrue_b32(),values+6*nbValuesInVec,input6_val);
    svst1_s32(svptrue_b32(),values+7*nbValuesInVec,input7_val);
    svst1_s32(svptrue_b32(),values+8*nbValuesInVec,input8_val);
    svst1_s32(svptrue_b32(),values+9*nbValuesInVec,input9_val);
    svst1_s32(svptrue_b32(),values+10*nbValuesInVec,input10_val);
    svst1_s32(svptrue_b32(),values+11*nbValuesInVec,input11_val);
}


inline void CoreSmallSort13( int* __restrict__ ptr1, int* __restrict__ values){
    const int nbValuesInVec = svcntw();
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+nbValuesInVec);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*nbValuesInVec);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*nbValuesInVec);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*nbValuesInVec);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*nbValuesInVec);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*nbValuesInVec);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*nbValuesInVec);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr1+8*nbValuesInVec);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr1+9*nbValuesInVec);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr1+10*nbValuesInVec);
    svint32_t input11 = svld1_s32(svptrue_b32(),ptr1+11*nbValuesInVec);
    svint32_t input12 = svld1_s32(svptrue_b32(),ptr1+12*nbValuesInVec);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
    svint32_t input8_val = svld1_s32(svptrue_b32(),values+8*nbValuesInVec);
    svint32_t input9_val = svld1_s32(svptrue_b32(),values+9*nbValuesInVec);
    svint32_t input10_val = svld1_s32(svptrue_b32(),values+10*nbValuesInVec);
    svint32_t input11_val = svld1_s32(svptrue_b32(),values+11*nbValuesInVec);
    svint32_t input12_val = svld1_s32(svptrue_b32(),values+12*nbValuesInVec);
    CoreSmallSort13(input0,input1,input2,input3,
                    input4,input5,input6,input7,
                    input8,input9,input10,input11,
                    input12,input0_val,input1_val,input2_val,input3_val,
                    input4_val,input5_val,input6_val,input7_val,
                    input8_val,input9_val,input10_val,input11_val,
                    input12_val);
    svst1_s32(svptrue_b32(),ptr1, input0);
    svst1_s32(svptrue_b32(),ptr1+nbValuesInVec,input1);
    svst1_s32(svptrue_b32(),ptr1+2*nbValuesInVec,input2);
    svst1_s32(svptrue_b32(),ptr1+3*nbValuesInVec,input3);
    svst1_s32(svptrue_b32(),ptr1+4*nbValuesInVec,input4);
    svst1_s32(svptrue_b32(),ptr1+5*nbValuesInVec,input5);
    svst1_s32(svptrue_b32(),ptr1+6*nbValuesInVec,input6);
    svst1_s32(svptrue_b32(),ptr1+7*nbValuesInVec,input7);
    svst1_s32(svptrue_b32(),ptr1+8*nbValuesInVec,input8);
    svst1_s32(svptrue_b32(),ptr1+9*nbValuesInVec,input9);
    svst1_s32(svptrue_b32(),ptr1+10*nbValuesInVec,input10);
    svst1_s32(svptrue_b32(),ptr1+11*nbValuesInVec,input11);
    svst1_s32(svptrue_b32(),ptr1+12*nbValuesInVec,input12);
    svst1_s32(svptrue_b32(),values, input0_val);
    svst1_s32(svptrue_b32(),values+nbValuesInVec,input1_val);
    svst1_s32(svptrue_b32(),values+2*nbValuesInVec,input2_val);
    svst1_s32(svptrue_b32(),values+3*nbValuesInVec,input3_val);
    svst1_s32(svptrue_b32(),values+4*nbValuesInVec,input4_val);
    svst1_s32(svptrue_b32(),values+5*nbValuesInVec,input5_val);
    svst1_s32(svptrue_b32(),values+6*nbValuesInVec,input6_val);
    svst1_s32(svptrue_b32(),values+7*nbValuesInVec,input7_val);
    svst1_s32(svptrue_b32(),values+8*nbValuesInVec,input8_val);
    svst1_s32(svptrue_b32(),values+9*nbValuesInVec,input9_val);
    svst1_s32(svptrue_b32(),values+10*nbValuesInVec,input10_val);
    svst1_s32(svptrue_b32(),values+11*nbValuesInVec,input11_val);
    svst1_s32(svptrue_b32(),values+12*nbValuesInVec,input12_val);
}


inline void CoreSmallSort14( int* __restrict__ ptr1, int* __restrict__ values){
    const int nbValuesInVec = svcntw();
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+nbValuesInVec);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*nbValuesInVec);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*nbValuesInVec);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*nbValuesInVec);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*nbValuesInVec);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*nbValuesInVec);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*nbValuesInVec);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr1+8*nbValuesInVec);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr1+9*nbValuesInVec);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr1+10*nbValuesInVec);
    svint32_t input11 = svld1_s32(svptrue_b32(),ptr1+11*nbValuesInVec);
    svint32_t input12 = svld1_s32(svptrue_b32(),ptr1+12*nbValuesInVec);
    svint32_t input13 = svld1_s32(svptrue_b32(),ptr1+13*nbValuesInVec);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
    svint32_t input8_val = svld1_s32(svptrue_b32(),values+8*nbValuesInVec);
    svint32_t input9_val = svld1_s32(svptrue_b32(),values+9*nbValuesInVec);
    svint32_t input10_val = svld1_s32(svptrue_b32(),values+10*nbValuesInVec);
    svint32_t input11_val = svld1_s32(svptrue_b32(),values+11*nbValuesInVec);
    svint32_t input12_val = svld1_s32(svptrue_b32(),values+12*nbValuesInVec);
    svint32_t input13_val = svld1_s32(svptrue_b32(),values+13*nbValuesInVec);
    CoreSmallSort14(input0,input1,input2,input3,
                    input4,input5,input6,input7,
                    input8,input9,input10,input11,
                    input12,input13,input0_val,input1_val,input2_val,input3_val,
                    input4_val,input5_val,input6_val,input7_val,
                    input8_val,input9_val,input10_val,input11_val,
                    input12_val,input13_val);
    svst1_s32(svptrue_b32(),ptr1, input0);
    svst1_s32(svptrue_b32(),ptr1+nbValuesInVec,input1);
    svst1_s32(svptrue_b32(),ptr1+2*nbValuesInVec,input2);
    svst1_s32(svptrue_b32(),ptr1+3*nbValuesInVec,input3);
    svst1_s32(svptrue_b32(),ptr1+4*nbValuesInVec,input4);
    svst1_s32(svptrue_b32(),ptr1+5*nbValuesInVec,input5);
    svst1_s32(svptrue_b32(),ptr1+6*nbValuesInVec,input6);
    svst1_s32(svptrue_b32(),ptr1+7*nbValuesInVec,input7);
    svst1_s32(svptrue_b32(),ptr1+8*nbValuesInVec,input8);
    svst1_s32(svptrue_b32(),ptr1+9*nbValuesInVec,input9);
    svst1_s32(svptrue_b32(),ptr1+10*nbValuesInVec,input10);
    svst1_s32(svptrue_b32(),ptr1+11*nbValuesInVec,input11);
    svst1_s32(svptrue_b32(),ptr1+12*nbValuesInVec,input12);
    svst1_s32(svptrue_b32(),ptr1+13*nbValuesInVec,input13);
    svst1_s32(svptrue_b32(),values, input0_val);
    svst1_s32(svptrue_b32(),values+nbValuesInVec,input1_val);
    svst1_s32(svptrue_b32(),values+2*nbValuesInVec,input2_val);
    svst1_s32(svptrue_b32(),values+3*nbValuesInVec,input3_val);
    svst1_s32(svptrue_b32(),values+4*nbValuesInVec,input4_val);
    svst1_s32(svptrue_b32(),values+5*nbValuesInVec,input5_val);
    svst1_s32(svptrue_b32(),values+6*nbValuesInVec,input6_val);
    svst1_s32(svptrue_b32(),values+7*nbValuesInVec,input7_val);
    svst1_s32(svptrue_b32(),values+8*nbValuesInVec,input8_val);
    svst1_s32(svptrue_b32(),values+9*nbValuesInVec,input9_val);
    svst1_s32(svptrue_b32(),values+10*nbValuesInVec,input10_val);
    svst1_s32(svptrue_b32(),values+11*nbValuesInVec,input11_val);
    svst1_s32(svptrue_b32(),values+12*nbValuesInVec,input12_val);
    svst1_s32(svptrue_b32(),values+13*nbValuesInVec,input13_val);
}


inline void CoreSmallSort15( int* __restrict__ ptr1, int* __restrict__ values){
    const int nbValuesInVec = svcntw();
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+nbValuesInVec);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*nbValuesInVec);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*nbValuesInVec);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*nbValuesInVec);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*nbValuesInVec);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*nbValuesInVec);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*nbValuesInVec);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr1+8*nbValuesInVec);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr1+9*nbValuesInVec);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr1+10*nbValuesInVec);
    svint32_t input11 = svld1_s32(svptrue_b32(),ptr1+11*nbValuesInVec);
    svint32_t input12 = svld1_s32(svptrue_b32(),ptr1+12*nbValuesInVec);
    svint32_t input13 = svld1_s32(svptrue_b32(),ptr1+13*nbValuesInVec);
    svint32_t input14 = svld1_s32(svptrue_b32(),ptr1+14*nbValuesInVec);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
    svint32_t input8_val = svld1_s32(svptrue_b32(),values+8*nbValuesInVec);
    svint32_t input9_val = svld1_s32(svptrue_b32(),values+9*nbValuesInVec);
    svint32_t input10_val = svld1_s32(svptrue_b32(),values+10*nbValuesInVec);
    svint32_t input11_val = svld1_s32(svptrue_b32(),values+11*nbValuesInVec);
    svint32_t input12_val = svld1_s32(svptrue_b32(),values+12*nbValuesInVec);
    svint32_t input13_val = svld1_s32(svptrue_b32(),values+13*nbValuesInVec);
    svint32_t input14_val = svld1_s32(svptrue_b32(),values+14*nbValuesInVec);
    CoreSmallSort15(input0,input1,input2,input3,
                    input4,input5,input6,input7,
                    input8,input9,input10,input11,
                    input12,input13,input14,input0_val,input1_val,input2_val,input3_val,
                    input4_val,input5_val,input6_val,input7_val,
                    input8_val,input9_val,input10_val,input11_val,
                    input12_val,input13_val,input14_val);
    svst1_s32(svptrue_b32(),ptr1, input0);
    svst1_s32(svptrue_b32(),ptr1+nbValuesInVec,input1);
    svst1_s32(svptrue_b32(),ptr1+2*nbValuesInVec,input2);
    svst1_s32(svptrue_b32(),ptr1+3*nbValuesInVec,input3);
    svst1_s32(svptrue_b32(),ptr1+4*nbValuesInVec,input4);
    svst1_s32(svptrue_b32(),ptr1+5*nbValuesInVec,input5);
    svst1_s32(svptrue_b32(),ptr1+6*nbValuesInVec,input6);
    svst1_s32(svptrue_b32(),ptr1+7*nbValuesInVec,input7);
    svst1_s32(svptrue_b32(),ptr1+8*nbValuesInVec,input8);
    svst1_s32(svptrue_b32(),ptr1+9*nbValuesInVec,input9);
    svst1_s32(svptrue_b32(),ptr1+10*nbValuesInVec,input10);
    svst1_s32(svptrue_b32(),ptr1+11*nbValuesInVec,input11);
    svst1_s32(svptrue_b32(),ptr1+12*nbValuesInVec,input12);
    svst1_s32(svptrue_b32(),ptr1+13*nbValuesInVec,input13);
    svst1_s32(svptrue_b32(),ptr1+14*nbValuesInVec,input14);
    svst1_s32(svptrue_b32(),values, input0_val);
    svst1_s32(svptrue_b32(),values+nbValuesInVec,input1_val);
    svst1_s32(svptrue_b32(),values+2*nbValuesInVec,input2_val);
    svst1_s32(svptrue_b32(),values+3*nbValuesInVec,input3_val);
    svst1_s32(svptrue_b32(),values+4*nbValuesInVec,input4_val);
    svst1_s32(svptrue_b32(),values+5*nbValuesInVec,input5_val);
    svst1_s32(svptrue_b32(),values+6*nbValuesInVec,input6_val);
    svst1_s32(svptrue_b32(),values+7*nbValuesInVec,input7_val);
    svst1_s32(svptrue_b32(),values+8*nbValuesInVec,input8_val);
    svst1_s32(svptrue_b32(),values+9*nbValuesInVec,input9_val);
    svst1_s32(svptrue_b32(),values+10*nbValuesInVec,input10_val);
    svst1_s32(svptrue_b32(),values+11*nbValuesInVec,input11_val);
    svst1_s32(svptrue_b32(),values+12*nbValuesInVec,input12_val);
    svst1_s32(svptrue_b32(),values+13*nbValuesInVec,input13_val);
    svst1_s32(svptrue_b32(),values+14*nbValuesInVec,input14_val);
}


inline void CoreSmallSort16( int* __restrict__ ptr1, int* __restrict__ values){
    const int nbValuesInVec = svcntw();
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+nbValuesInVec);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*nbValuesInVec);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*nbValuesInVec);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*nbValuesInVec);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*nbValuesInVec);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*nbValuesInVec);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*nbValuesInVec);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr1+8*nbValuesInVec);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr1+9*nbValuesInVec);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr1+10*nbValuesInVec);
    svint32_t input11 = svld1_s32(svptrue_b32(),ptr1+11*nbValuesInVec);
    svint32_t input12 = svld1_s32(svptrue_b32(),ptr1+12*nbValuesInVec);
    svint32_t input13 = svld1_s32(svptrue_b32(),ptr1+13*nbValuesInVec);
    svint32_t input14 = svld1_s32(svptrue_b32(),ptr1+14*nbValuesInVec);
    svint32_t input15 = svld1_s32(svptrue_b32(),ptr1+15*nbValuesInVec);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
    svint32_t input8_val = svld1_s32(svptrue_b32(),values+8*nbValuesInVec);
    svint32_t input9_val = svld1_s32(svptrue_b32(),values+9*nbValuesInVec);
    svint32_t input10_val = svld1_s32(svptrue_b32(),values+10*nbValuesInVec);
    svint32_t input11_val = svld1_s32(svptrue_b32(),values+11*nbValuesInVec);
    svint32_t input12_val = svld1_s32(svptrue_b32(),values+12*nbValuesInVec);
    svint32_t input13_val = svld1_s32(svptrue_b32(),values+13*nbValuesInVec);
    svint32_t input14_val = svld1_s32(svptrue_b32(),values+14*nbValuesInVec);
    svint32_t input15_val = svld1_s32(svptrue_b32(),values+15*nbValuesInVec);
    CoreSmallSort16(input0,input1,input2,input3,
                    input4,input5,input6,input7,
                    input8,input9,input10,input11,
                    input12,input13,input14,input15,input0_val,input1_val,input2_val,input3_val,
                    input4_val,input5_val,input6_val,input7_val,
                    input8_val,input9_val,input10_val,input11_val,
                    input12_val,input13_val,input14_val,input15_val);
    svst1_s32(svptrue_b32(),ptr1, input0);
    svst1_s32(svptrue_b32(),ptr1+nbValuesInVec,input1);
    svst1_s32(svptrue_b32(),ptr1+2*nbValuesInVec,input2);
    svst1_s32(svptrue_b32(),ptr1+3*nbValuesInVec,input3);
    svst1_s32(svptrue_b32(),ptr1+4*nbValuesInVec,input4);
    svst1_s32(svptrue_b32(),ptr1+5*nbValuesInVec,input5);
    svst1_s32(svptrue_b32(),ptr1+6*nbValuesInVec,input6);
    svst1_s32(svptrue_b32(),ptr1+7*nbValuesInVec,input7);
    svst1_s32(svptrue_b32(),ptr1+8*nbValuesInVec,input8);
    svst1_s32(svptrue_b32(),ptr1+9*nbValuesInVec,input9);
    svst1_s32(svptrue_b32(),ptr1+10*nbValuesInVec,input10);
    svst1_s32(svptrue_b32(),ptr1+11*nbValuesInVec,input11);
    svst1_s32(svptrue_b32(),ptr1+12*nbValuesInVec,input12);
    svst1_s32(svptrue_b32(),ptr1+13*nbValuesInVec,input13);
    svst1_s32(svptrue_b32(),ptr1+14*nbValuesInVec,input14);
    svst1_s32(svptrue_b32(),ptr1+15*nbValuesInVec,input15);
    svst1_s32(svptrue_b32(),values, input0_val);
    svst1_s32(svptrue_b32(),values+nbValuesInVec,input1_val);
    svst1_s32(svptrue_b32(),values+2*nbValuesInVec,input2_val);
    svst1_s32(svptrue_b32(),values+3*nbValuesInVec,input3_val);
    svst1_s32(svptrue_b32(),values+4*nbValuesInVec,input4_val);
    svst1_s32(svptrue_b32(),values+5*nbValuesInVec,input5_val);
    svst1_s32(svptrue_b32(),values+6*nbValuesInVec,input6_val);
    svst1_s32(svptrue_b32(),values+7*nbValuesInVec,input7_val);
    svst1_s32(svptrue_b32(),values+8*nbValuesInVec,input8_val);
    svst1_s32(svptrue_b32(),values+9*nbValuesInVec,input9_val);
    svst1_s32(svptrue_b32(),values+10*nbValuesInVec,input10_val);
    svst1_s32(svptrue_b32(),values+11*nbValuesInVec,input11_val);
    svst1_s32(svptrue_b32(),values+12*nbValuesInVec,input12_val);
    svst1_s32(svptrue_b32(),values+13*nbValuesInVec,input13_val);
    svst1_s32(svptrue_b32(),values+14*nbValuesInVec,input14_val);
    svst1_s32(svptrue_b32(),values+15*nbValuesInVec,input15_val);
}


inline void CoreSmallSort( std::pair<int,int> ptr1[]){
    const int nbValuesInVec = svcntw();
    svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
    svint32_t input0_v0 = svget2_s32(input0, 0);
    svint32_t input0_v1 = svget2_s32(input0, 1);
    CoreSmallSort(input0_v0,
    input0_v1);
    svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
}
inline void CoreSmallSort2( std::pair<int,int> ptr1[]){
    const int nbValuesInVec = svcntw();
    svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
    svint32_t input0_v0 = svget2_s32(input0, 0);
    svint32_t input0_v1 = svget2_s32(input0, 1);
    svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
    svint32_t input1_v0 = svget2_s32(input1, 0);
    svint32_t input1_v1 = svget2_s32(input1, 1);
    CoreSmallSort2(input0_v0, input1_v0,
    input0_v1, input1_v1);
    svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));

}
inline void CoreSmallSort3( std::pair<int,int> ptr1[]){
    const int nbValuesInVec = svcntw();
    svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
    svint32_t input0_v0 = svget2_s32(input0, 0);
    svint32_t input0_v1 = svget2_s32(input0, 1);
    svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
    svint32_t input1_v0 = svget2_s32(input1, 0);
    svint32_t input1_v1 = svget2_s32(input1, 1);
    svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
    svint32_t input2_v0 = svget2_s32(input2, 0);
    svint32_t input2_v1 = svget2_s32(input2, 1);
    CoreSmallSort3(input0_v0, input1_v0, input2_v0,
    input0_v1, input1_v1, input2_v1);
    svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
}
inline void CoreSmallSort4( std::pair<int,int> ptr1[]){
    const int nbValuesInVec = svcntw();
    svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
    svint32_t input0_v0 = svget2_s32(input0, 0);
    svint32_t input0_v1 = svget2_s32(input0, 1);
    svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
    svint32_t input1_v0 = svget2_s32(input1, 0);
    svint32_t input1_v1 = svget2_s32(input1, 1);
    svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
    svint32_t input2_v0 = svget2_s32(input2, 0);
    svint32_t input2_v1 = svget2_s32(input2, 1);
    svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
    svint32_t input3_v0 = svget2_s32(input3, 0);
    svint32_t input3_v1 = svget2_s32(input3, 1);
    CoreSmallSort4(input0_v0, input1_v0, input2_v0, input3_v0,
    input0_v1, input1_v1, input2_v1, input3_v1);
    svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
}
inline void CoreSmallSort5( std::pair<int,int> ptr1[]){
    const int nbValuesInVec = svcntw();
    svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
    svint32_t input0_v0 = svget2_s32(input0, 0);
    svint32_t input0_v1 = svget2_s32(input0, 1);
    svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
    svint32_t input1_v0 = svget2_s32(input1, 0);
    svint32_t input1_v1 = svget2_s32(input1, 1);
    svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
    svint32_t input2_v0 = svget2_s32(input2, 0);
    svint32_t input2_v1 = svget2_s32(input2, 1);
    svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
    svint32_t input3_v0 = svget2_s32(input3, 0);
    svint32_t input3_v1 = svget2_s32(input3, 1);
    svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
    svint32_t input4_v0 = svget2_s32(input4, 0);
    svint32_t input4_v1 = svget2_s32(input4, 1);
    CoreSmallSort5(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0,
    input0_v1, input1_v1, input2_v1, input3_v1, input4_v1);
    svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
}
inline void CoreSmallSort6( std::pair<int,int> ptr1[]){
    const int nbValuesInVec = svcntw();
    svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
    svint32_t input0_v0 = svget2_s32(input0, 0);
    svint32_t input0_v1 = svget2_s32(input0, 1);
    svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
    svint32_t input1_v0 = svget2_s32(input1, 0);
    svint32_t input1_v1 = svget2_s32(input1, 1);
    svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
    svint32_t input2_v0 = svget2_s32(input2, 0);
    svint32_t input2_v1 = svget2_s32(input2, 1);
    svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
    svint32_t input3_v0 = svget2_s32(input3, 0);
    svint32_t input3_v1 = svget2_s32(input3, 1);
    svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
    svint32_t input4_v0 = svget2_s32(input4, 0);
    svint32_t input4_v1 = svget2_s32(input4, 1);
    svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
    svint32_t input5_v0 = svget2_s32(input5, 0);
    svint32_t input5_v1 = svget2_s32(input5, 1);
    CoreSmallSort6(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0,
    input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1);
    svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));

}
inline void CoreSmallSort7( std::pair<int,int> ptr1[]){
    const int nbValuesInVec = svcntw();
    svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
    svint32_t input0_v0 = svget2_s32(input0, 0);
    svint32_t input0_v1 = svget2_s32(input0, 1);
    svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
    svint32_t input1_v0 = svget2_s32(input1, 0);
    svint32_t input1_v1 = svget2_s32(input1, 1);
    svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
    svint32_t input2_v0 = svget2_s32(input2, 0);
    svint32_t input2_v1 = svget2_s32(input2, 1);
    svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
    svint32_t input3_v0 = svget2_s32(input3, 0);
    svint32_t input3_v1 = svget2_s32(input3, 1);
    svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
    svint32_t input4_v0 = svget2_s32(input4, 0);
    svint32_t input4_v1 = svget2_s32(input4, 1);
    svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
    svint32_t input5_v0 = svget2_s32(input5, 0);
    svint32_t input5_v1 = svget2_s32(input5, 1);
    svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
    svint32_t input6_v0 = svget2_s32(input6, 0);
    svint32_t input6_v1 = svget2_s32(input6, 1);
    CoreSmallSort7(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0,
    input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1);
    svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));

}
inline void CoreSmallSort8( std::pair<int,int> ptr1[]){
    const int nbValuesInVec = svcntw();
    svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
    svint32_t input0_v0 = svget2_s32(input0, 0);
    svint32_t input0_v1 = svget2_s32(input0, 1);
    svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
    svint32_t input1_v0 = svget2_s32(input1, 0);
    svint32_t input1_v1 = svget2_s32(input1, 1);
    svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
    svint32_t input2_v0 = svget2_s32(input2, 0);
    svint32_t input2_v1 = svget2_s32(input2, 1);
    svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
    svint32_t input3_v0 = svget2_s32(input3, 0);
    svint32_t input3_v1 = svget2_s32(input3, 1);
    svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
    svint32_t input4_v0 = svget2_s32(input4, 0);
    svint32_t input4_v1 = svget2_s32(input4, 1);
    svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
    svint32_t input5_v0 = svget2_s32(input5, 0);
    svint32_t input5_v1 = svget2_s32(input5, 1);
    svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
    svint32_t input6_v0 = svget2_s32(input6, 0);
    svint32_t input6_v1 = svget2_s32(input6, 1);
    svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
    svint32_t input7_v0 = svget2_s32(input7, 0);
    svint32_t input7_v1 = svget2_s32(input7, 1);
    CoreSmallSort8(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0,
    input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1);
    svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
}
inline void CoreSmallSort9( std::pair<int,int> ptr1[]){
    const int nbValuesInVec = svcntw();
    svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
    svint32_t input0_v0 = svget2_s32(input0, 0);
    svint32_t input0_v1 = svget2_s32(input0, 1);
    svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
    svint32_t input1_v0 = svget2_s32(input1, 0);
    svint32_t input1_v1 = svget2_s32(input1, 1);
    svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
    svint32_t input2_v0 = svget2_s32(input2, 0);
    svint32_t input2_v1 = svget2_s32(input2, 1);
    svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
    svint32_t input3_v0 = svget2_s32(input3, 0);
    svint32_t input3_v1 = svget2_s32(input3, 1);
    svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
    svint32_t input4_v0 = svget2_s32(input4, 0);
    svint32_t input4_v1 = svget2_s32(input4, 1);
    svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
    svint32_t input5_v0 = svget2_s32(input5, 0);
    svint32_t input5_v1 = svget2_s32(input5, 1);
    svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
    svint32_t input6_v0 = svget2_s32(input6, 0);
    svint32_t input6_v1 = svget2_s32(input6, 1);
    svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
    svint32_t input7_v0 = svget2_s32(input7, 0);
    svint32_t input7_v1 = svget2_s32(input7, 1);
    svint32x2_t input8 = svld2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec));
    svint32_t input8_v0 = svget2_s32(input8, 0);
    svint32_t input8_v1 = svget2_s32(input8, 1);
    CoreSmallSort9(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0, input8_v0,
    input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1, input8_v1);
    svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec), svcreate2_s32(input8_v0,input8_v1));

}
inline void CoreSmallSort10( std::pair<int,int> ptr1[]){
    const int nbValuesInVec = svcntw();
    svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
    svint32_t input0_v0 = svget2_s32(input0, 0);
    svint32_t input0_v1 = svget2_s32(input0, 1);
    svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
    svint32_t input1_v0 = svget2_s32(input1, 0);
    svint32_t input1_v1 = svget2_s32(input1, 1);
    svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
    svint32_t input2_v0 = svget2_s32(input2, 0);
    svint32_t input2_v1 = svget2_s32(input2, 1);
    svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
    svint32_t input3_v0 = svget2_s32(input3, 0);
    svint32_t input3_v1 = svget2_s32(input3, 1);
    svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
    svint32_t input4_v0 = svget2_s32(input4, 0);
    svint32_t input4_v1 = svget2_s32(input4, 1);
    svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
    svint32_t input5_v0 = svget2_s32(input5, 0);
    svint32_t input5_v1 = svget2_s32(input5, 1);
    svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
    svint32_t input6_v0 = svget2_s32(input6, 0);
    svint32_t input6_v1 = svget2_s32(input6, 1);
    svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
    svint32_t input7_v0 = svget2_s32(input7, 0);
    svint32_t input7_v1 = svget2_s32(input7, 1);
    svint32x2_t input8 = svld2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec));
    svint32_t input8_v0 = svget2_s32(input8, 0);
    svint32_t input8_v1 = svget2_s32(input8, 1);
    svint32x2_t input9 = svld2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec));
    svint32_t input9_v0 = svget2_s32(input9, 0);
    svint32_t input9_v1 = svget2_s32(input9, 1);
    CoreSmallSort10(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0, input8_v0, input9_v0,
    input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1, input8_v1, input9_v1);
    svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec), svcreate2_s32(input8_v0,input8_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec), svcreate2_s32(input9_v0,input9_v1));
}
inline void CoreSmallSort11( std::pair<int,int> ptr1[]){
    const int nbValuesInVec = svcntw();
    svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
    svint32_t input0_v0 = svget2_s32(input0, 0);
    svint32_t input0_v1 = svget2_s32(input0, 1);
    svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
    svint32_t input1_v0 = svget2_s32(input1, 0);
    svint32_t input1_v1 = svget2_s32(input1, 1);
    svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
    svint32_t input2_v0 = svget2_s32(input2, 0);
    svint32_t input2_v1 = svget2_s32(input2, 1);
    svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
    svint32_t input3_v0 = svget2_s32(input3, 0);
    svint32_t input3_v1 = svget2_s32(input3, 1);
    svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
    svint32_t input4_v0 = svget2_s32(input4, 0);
    svint32_t input4_v1 = svget2_s32(input4, 1);
    svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
    svint32_t input5_v0 = svget2_s32(input5, 0);
    svint32_t input5_v1 = svget2_s32(input5, 1);
    svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
    svint32_t input6_v0 = svget2_s32(input6, 0);
    svint32_t input6_v1 = svget2_s32(input6, 1);
    svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
    svint32_t input7_v0 = svget2_s32(input7, 0);
    svint32_t input7_v1 = svget2_s32(input7, 1);
    svint32x2_t input8 = svld2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec));
    svint32_t input8_v0 = svget2_s32(input8, 0);
    svint32_t input8_v1 = svget2_s32(input8, 1);
    svint32x2_t input9 = svld2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec));
    svint32_t input9_v0 = svget2_s32(input9, 0);
    svint32_t input9_v1 = svget2_s32(input9, 1);
    svint32x2_t input10 = svld2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec));
    svint32_t input10_v0 = svget2_s32(input10, 0);
    svint32_t input10_v1 = svget2_s32(input10, 1);
    CoreSmallSort11(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0, input8_v0, input9_v0, input10_v0,
    input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1, input8_v1, input9_v1, input10_v1);
    svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec), svcreate2_s32(input8_v0,input8_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec), svcreate2_s32(input9_v0,input9_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec), svcreate2_s32(input10_v0,input10_v1));
}
inline void CoreSmallSort12( std::pair<int,int> ptr1[]){
    const int nbValuesInVec = svcntw();
    svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
    svint32_t input0_v0 = svget2_s32(input0, 0);
    svint32_t input0_v1 = svget2_s32(input0, 1);
    svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
    svint32_t input1_v0 = svget2_s32(input1, 0);
    svint32_t input1_v1 = svget2_s32(input1, 1);
    svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
    svint32_t input2_v0 = svget2_s32(input2, 0);
    svint32_t input2_v1 = svget2_s32(input2, 1);
    svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
    svint32_t input3_v0 = svget2_s32(input3, 0);
    svint32_t input3_v1 = svget2_s32(input3, 1);
    svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
    svint32_t input4_v0 = svget2_s32(input4, 0);
    svint32_t input4_v1 = svget2_s32(input4, 1);
    svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
    svint32_t input5_v0 = svget2_s32(input5, 0);
    svint32_t input5_v1 = svget2_s32(input5, 1);
    svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
    svint32_t input6_v0 = svget2_s32(input6, 0);
    svint32_t input6_v1 = svget2_s32(input6, 1);
    svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
    svint32_t input7_v0 = svget2_s32(input7, 0);
    svint32_t input7_v1 = svget2_s32(input7, 1);
    svint32x2_t input8 = svld2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec));
    svint32_t input8_v0 = svget2_s32(input8, 0);
    svint32_t input8_v1 = svget2_s32(input8, 1);
    svint32x2_t input9 = svld2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec));
    svint32_t input9_v0 = svget2_s32(input9, 0);
    svint32_t input9_v1 = svget2_s32(input9, 1);
    svint32x2_t input10 = svld2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec));
    svint32_t input10_v0 = svget2_s32(input10, 0);
    svint32_t input10_v1 = svget2_s32(input10, 1);
    svint32x2_t input11 = svld2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec));
    svint32_t input11_v0 = svget2_s32(input11, 0);
    svint32_t input11_v1 = svget2_s32(input11, 1);
    CoreSmallSort12(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0, input8_v0, input9_v0, input10_v0, input11_v0,
    input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1, input8_v1, input9_v1, input10_v1, input11_v1);
    svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec), svcreate2_s32(input8_v0,input8_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec), svcreate2_s32(input9_v0,input9_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec), svcreate2_s32(input10_v0,input10_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec), svcreate2_s32(input11_v0,input11_v1));

}
inline void CoreSmallSort13( std::pair<int,int> ptr1[]){
    const int nbValuesInVec = svcntw();
    svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
    svint32_t input0_v0 = svget2_s32(input0, 0);
    svint32_t input0_v1 = svget2_s32(input0, 1);
    svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
    svint32_t input1_v0 = svget2_s32(input1, 0);
    svint32_t input1_v1 = svget2_s32(input1, 1);
    svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
    svint32_t input2_v0 = svget2_s32(input2, 0);
    svint32_t input2_v1 = svget2_s32(input2, 1);
    svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
    svint32_t input3_v0 = svget2_s32(input3, 0);
    svint32_t input3_v1 = svget2_s32(input3, 1);
    svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
    svint32_t input4_v0 = svget2_s32(input4, 0);
    svint32_t input4_v1 = svget2_s32(input4, 1);
    svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
    svint32_t input5_v0 = svget2_s32(input5, 0);
    svint32_t input5_v1 = svget2_s32(input5, 1);
    svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
    svint32_t input6_v0 = svget2_s32(input6, 0);
    svint32_t input6_v1 = svget2_s32(input6, 1);
    svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
    svint32_t input7_v0 = svget2_s32(input7, 0);
    svint32_t input7_v1 = svget2_s32(input7, 1);
    svint32x2_t input8 = svld2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec));
    svint32_t input8_v0 = svget2_s32(input8, 0);
    svint32_t input8_v1 = svget2_s32(input8, 1);
    svint32x2_t input9 = svld2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec));
    svint32_t input9_v0 = svget2_s32(input9, 0);
    svint32_t input9_v1 = svget2_s32(input9, 1);
    svint32x2_t input10 = svld2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec));
    svint32_t input10_v0 = svget2_s32(input10, 0);
    svint32_t input10_v1 = svget2_s32(input10, 1);
    svint32x2_t input11 = svld2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec));
    svint32_t input11_v0 = svget2_s32(input11, 0);
    svint32_t input11_v1 = svget2_s32(input11, 1);
    svint32x2_t input12 = svld2_s32(svptrue_b32(),(int*)(ptr1+12*nbValuesInVec));
    svint32_t input12_v0 = svget2_s32(input12, 0);
    svint32_t input12_v1 = svget2_s32(input12, 1);
    CoreSmallSort13(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0, input8_v0, input9_v0, input10_v0, input11_v0, input12_v0,
    input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1, input8_v1, input9_v1, input10_v1, input11_v1, input12_v1);
    svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec), svcreate2_s32(input8_v0,input8_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec), svcreate2_s32(input9_v0,input9_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec), svcreate2_s32(input10_v0,input10_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec), svcreate2_s32(input11_v0,input11_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+12*nbValuesInVec), svcreate2_s32(input12_v0,input12_v1));
}
inline void CoreSmallSort14( std::pair<int,int> ptr1[]){
    const int nbValuesInVec = svcntw();
    svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
    svint32_t input0_v0 = svget2_s32(input0, 0);
    svint32_t input0_v1 = svget2_s32(input0, 1);
    svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
    svint32_t input1_v0 = svget2_s32(input1, 0);
    svint32_t input1_v1 = svget2_s32(input1, 1);
    svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
    svint32_t input2_v0 = svget2_s32(input2, 0);
    svint32_t input2_v1 = svget2_s32(input2, 1);
    svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
    svint32_t input3_v0 = svget2_s32(input3, 0);
    svint32_t input3_v1 = svget2_s32(input3, 1);
    svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
    svint32_t input4_v0 = svget2_s32(input4, 0);
    svint32_t input4_v1 = svget2_s32(input4, 1);
    svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
    svint32_t input5_v0 = svget2_s32(input5, 0);
    svint32_t input5_v1 = svget2_s32(input5, 1);
    svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
    svint32_t input6_v0 = svget2_s32(input6, 0);
    svint32_t input6_v1 = svget2_s32(input6, 1);
    svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
    svint32_t input7_v0 = svget2_s32(input7, 0);
    svint32_t input7_v1 = svget2_s32(input7, 1);
    svint32x2_t input8 = svld2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec));
    svint32_t input8_v0 = svget2_s32(input8, 0);
    svint32_t input8_v1 = svget2_s32(input8, 1);
    svint32x2_t input9 = svld2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec));
    svint32_t input9_v0 = svget2_s32(input9, 0);
    svint32_t input9_v1 = svget2_s32(input9, 1);
    svint32x2_t input10 = svld2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec));
    svint32_t input10_v0 = svget2_s32(input10, 0);
    svint32_t input10_v1 = svget2_s32(input10, 1);
    svint32x2_t input11 = svld2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec));
    svint32_t input11_v0 = svget2_s32(input11, 0);
    svint32_t input11_v1 = svget2_s32(input11, 1);
    svint32x2_t input12 = svld2_s32(svptrue_b32(),(int*)(ptr1+12*nbValuesInVec));
    svint32_t input12_v0 = svget2_s32(input12, 0);
    svint32_t input12_v1 = svget2_s32(input12, 1);
    svint32x2_t input13 = svld2_s32(svptrue_b32(),(int*)(ptr1+13*nbValuesInVec));
    svint32_t input13_v0 = svget2_s32(input13, 0);
    svint32_t input13_v1 = svget2_s32(input13, 1);
    CoreSmallSort14(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0, input8_v0, input9_v0, input10_v0, input11_v0, input12_v0, input13_v0,
    input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1, input8_v1, input9_v1, input10_v1, input11_v1, input12_v1, input13_v1);
    svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec), svcreate2_s32(input8_v0,input8_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec), svcreate2_s32(input9_v0,input9_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec), svcreate2_s32(input10_v0,input10_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec), svcreate2_s32(input11_v0,input11_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+12*nbValuesInVec), svcreate2_s32(input12_v0,input12_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+13*nbValuesInVec), svcreate2_s32(input13_v0,input13_v1));
}
inline void CoreSmallSort15( std::pair<int,int> ptr1[]){
    const int nbValuesInVec = svcntw();
    svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
    svint32_t input0_v0 = svget2_s32(input0, 0);
    svint32_t input0_v1 = svget2_s32(input0, 1);
    svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
    svint32_t input1_v0 = svget2_s32(input1, 0);
    svint32_t input1_v1 = svget2_s32(input1, 1);
    svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
    svint32_t input2_v0 = svget2_s32(input2, 0);
    svint32_t input2_v1 = svget2_s32(input2, 1);
    svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
    svint32_t input3_v0 = svget2_s32(input3, 0);
    svint32_t input3_v1 = svget2_s32(input3, 1);
    svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
    svint32_t input4_v0 = svget2_s32(input4, 0);
    svint32_t input4_v1 = svget2_s32(input4, 1);
    svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
    svint32_t input5_v0 = svget2_s32(input5, 0);
    svint32_t input5_v1 = svget2_s32(input5, 1);
    svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
    svint32_t input6_v0 = svget2_s32(input6, 0);
    svint32_t input6_v1 = svget2_s32(input6, 1);
    svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
    svint32_t input7_v0 = svget2_s32(input7, 0);
    svint32_t input7_v1 = svget2_s32(input7, 1);
    svint32x2_t input8 = svld2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec));
    svint32_t input8_v0 = svget2_s32(input8, 0);
    svint32_t input8_v1 = svget2_s32(input8, 1);
    svint32x2_t input9 = svld2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec));
    svint32_t input9_v0 = svget2_s32(input9, 0);
    svint32_t input9_v1 = svget2_s32(input9, 1);
    svint32x2_t input10 = svld2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec));
    svint32_t input10_v0 = svget2_s32(input10, 0);
    svint32_t input10_v1 = svget2_s32(input10, 1);
    svint32x2_t input11 = svld2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec));
    svint32_t input11_v0 = svget2_s32(input11, 0);
    svint32_t input11_v1 = svget2_s32(input11, 1);
    svint32x2_t input12 = svld2_s32(svptrue_b32(),(int*)(ptr1+12*nbValuesInVec));
    svint32_t input12_v0 = svget2_s32(input12, 0);
    svint32_t input12_v1 = svget2_s32(input12, 1);
    svint32x2_t input13 = svld2_s32(svptrue_b32(),(int*)(ptr1+13*nbValuesInVec));
    svint32_t input13_v0 = svget2_s32(input13, 0);
    svint32_t input13_v1 = svget2_s32(input13, 1);
    svint32x2_t input14 = svld2_s32(svptrue_b32(),(int*)(ptr1+14*nbValuesInVec));
    svint32_t input14_v0 = svget2_s32(input14, 0);
    svint32_t input14_v1 = svget2_s32(input14, 1);
    CoreSmallSort15(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0, input8_v0, input9_v0, input10_v0, input11_v0, input12_v0, input13_v0, input14_v0,
    input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1, input8_v1, input9_v1, input10_v1, input11_v1, input12_v1, input13_v1, input14_v1);
    svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec), svcreate2_s32(input8_v0,input8_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec), svcreate2_s32(input9_v0,input9_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec), svcreate2_s32(input10_v0,input10_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec), svcreate2_s32(input11_v0,input11_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+12*nbValuesInVec), svcreate2_s32(input12_v0,input12_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+13*nbValuesInVec), svcreate2_s32(input13_v0,input13_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+14*nbValuesInVec), svcreate2_s32(input14_v0,input14_v1));
}
inline void CoreSmallSort16( std::pair<int,int> ptr1[]){
    const int nbValuesInVec = svcntw();
    svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
    svint32_t input0_v0 = svget2_s32(input0, 0);
    svint32_t input0_v1 = svget2_s32(input0, 1);
    svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
    svint32_t input1_v0 = svget2_s32(input1, 0);
    svint32_t input1_v1 = svget2_s32(input1, 1);
    svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
    svint32_t input2_v0 = svget2_s32(input2, 0);
    svint32_t input2_v1 = svget2_s32(input2, 1);
    svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
    svint32_t input3_v0 = svget2_s32(input3, 0);
    svint32_t input3_v1 = svget2_s32(input3, 1);
    svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
    svint32_t input4_v0 = svget2_s32(input4, 0);
    svint32_t input4_v1 = svget2_s32(input4, 1);
    svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
    svint32_t input5_v0 = svget2_s32(input5, 0);
    svint32_t input5_v1 = svget2_s32(input5, 1);
    svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
    svint32_t input6_v0 = svget2_s32(input6, 0);
    svint32_t input6_v1 = svget2_s32(input6, 1);
    svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
    svint32_t input7_v0 = svget2_s32(input7, 0);
    svint32_t input7_v1 = svget2_s32(input7, 1);
    svint32x2_t input8 = svld2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec));
    svint32_t input8_v0 = svget2_s32(input8, 0);
    svint32_t input8_v1 = svget2_s32(input8, 1);
    svint32x2_t input9 = svld2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec));
    svint32_t input9_v0 = svget2_s32(input9, 0);
    svint32_t input9_v1 = svget2_s32(input9, 1);
    svint32x2_t input10 = svld2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec));
    svint32_t input10_v0 = svget2_s32(input10, 0);
    svint32_t input10_v1 = svget2_s32(input10, 1);
    svint32x2_t input11 = svld2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec));
    svint32_t input11_v0 = svget2_s32(input11, 0);
    svint32_t input11_v1 = svget2_s32(input11, 1);
    svint32x2_t input12 = svld2_s32(svptrue_b32(),(int*)(ptr1+12*nbValuesInVec));
    svint32_t input12_v0 = svget2_s32(input12, 0);
    svint32_t input12_v1 = svget2_s32(input12, 1);
    svint32x2_t input13 = svld2_s32(svptrue_b32(),(int*)(ptr1+13*nbValuesInVec));
    svint32_t input13_v0 = svget2_s32(input13, 0);
    svint32_t input13_v1 = svget2_s32(input13, 1);
    svint32x2_t input14 = svld2_s32(svptrue_b32(),(int*)(ptr1+14*nbValuesInVec));
    svint32_t input14_v0 = svget2_s32(input14, 0);
    svint32_t input14_v1 = svget2_s32(input14, 1);
    svint32x2_t input15 = svld2_s32(svptrue_b32(),(int*)(ptr1+15*nbValuesInVec));
    svint32_t input15_v0 = svget2_s32(input15, 0);
    svint32_t input15_v1 = svget2_s32(input15, 1);
    CoreSmallSort16(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0, input8_v0, input9_v0, input10_v0, input11_v0, input12_v0, input13_v0, input14_v0, input15_v0,
    input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1, input8_v1, input9_v1, input10_v1, input11_v1, input12_v1, input13_v1, input14_v1, input15_v1);
    svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec), svcreate2_s32(input8_v0,input8_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec), svcreate2_s32(input9_v0,input9_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec), svcreate2_s32(input10_v0,input10_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec), svcreate2_s32(input11_v0,input11_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+12*nbValuesInVec), svcreate2_s32(input12_v0,input12_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+13*nbValuesInVec), svcreate2_s32(input13_v0,input13_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+14*nbValuesInVec), svcreate2_s32(input14_v0,input14_v1));
    svst2_s32(svptrue_b32(),(int*)(ptr1+15*nbValuesInVec), svcreate2_s32(input15_v0,input15_v1));
}



////////////////////////////////////////////////////////////


inline void SmallSort16V(std::pair<int,int>* ptr1, const size_t length){
    // length is limited to 4 times size of a vec
    const int nbValuesInVec = svcntw();
    const int nbVecs = (length+nbValuesInVec-1)/nbValuesInVec;
    const int rest = nbVecs*nbValuesInVec-length;
    const int lastVecSize = nbValuesInVec-rest;

    const svint32_t intMaxVector = svdup_s32(INT_MAX);
    const svbool_t maskRest = SortSVE::getTrueFalseMask32(lastVecSize);

    switch(nbVecs){
    case 1:
    {
        svint32x2_t input0 = svld2_s32(maskRest,(int*)(ptr1));
        svint32_t input0_v0 = svget2_s32(input0, 0);
        svint32_t input0_v1 = svget2_s32(input0, 1);
        input0_v0 = svsel_s32(maskRest, input0_v0, intMaxVector);
        CoreSmallSort(input0_v0,
        input0_v1);
        svst2_s32(maskRest,(int*)(ptr1), svcreate2_s32(input0_v0,input0_v1));
    }
        break;
    case 2 :
    {
        svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
        svint32_t input0_v0 = svget2_s32(input0, 0);
        svint32_t input0_v1 = svget2_s32(input0, 1);
        svint32x2_t input1 = svld2_s32(maskRest,(int*)(ptr1+nbValuesInVec));
        svint32_t input1_v0 = svget2_s32(input1, 0);
        svint32_t input1_v1 = svget2_s32(input1, 1);
        input1_v0 = svsel_s32(maskRest, input1_v0, intMaxVector);
        CoreSmallSort2(input0_v0, input1_v0,
        input0_v1, input1_v1);
        svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
        svst2_s32(maskRest,(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
    break;
    }
    case 3 :
    {
        svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
        svint32_t input0_v0 = svget2_s32(input0, 0);
        svint32_t input0_v1 = svget2_s32(input0, 1);
        svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
        svint32_t input1_v0 = svget2_s32(input1, 0);
        svint32_t input1_v1 = svget2_s32(input1, 1);
        svint32x2_t input2 = svld2_s32(maskRest,(int*)(ptr1+2*nbValuesInVec));
        svint32_t input2_v0 = svget2_s32(input2, 0);
        svint32_t input2_v1 = svget2_s32(input2, 1);
        input2_v0 = svsel_s32(maskRest, input2_v0, intMaxVector);
        CoreSmallSort3(input0_v0, input1_v0, input2_v0,
        input0_v1, input1_v1, input2_v1);
        svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
        svst2_s32(maskRest,(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
    break;
    }
    case 4 :
    {
        svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
        svint32_t input0_v0 = svget2_s32(input0, 0);
        svint32_t input0_v1 = svget2_s32(input0, 1);
        svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
        svint32_t input1_v0 = svget2_s32(input1, 0);
        svint32_t input1_v1 = svget2_s32(input1, 1);
        svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
        svint32_t input2_v0 = svget2_s32(input2, 0);
        svint32_t input2_v1 = svget2_s32(input2, 1);
        svint32x2_t input3 = svld2_s32(maskRest,(int*)(ptr1+3*nbValuesInVec));
        svint32_t input3_v0 = svget2_s32(input3, 0);
        svint32_t input3_v1 = svget2_s32(input3, 1);
        input3_v0 = svsel_s32(maskRest, input3_v0, intMaxVector);
        CoreSmallSort4(input0_v0, input1_v0, input2_v0, input3_v0,
        input0_v1, input1_v1, input2_v1, input3_v1);
        svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
        svst2_s32(maskRest,(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
    break;
    }
    case 5 :
    {
        svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
        svint32_t input0_v0 = svget2_s32(input0, 0);
        svint32_t input0_v1 = svget2_s32(input0, 1);
        svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
        svint32_t input1_v0 = svget2_s32(input1, 0);
        svint32_t input1_v1 = svget2_s32(input1, 1);
        svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
        svint32_t input2_v0 = svget2_s32(input2, 0);
        svint32_t input2_v1 = svget2_s32(input2, 1);
        svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
        svint32_t input3_v0 = svget2_s32(input3, 0);
        svint32_t input3_v1 = svget2_s32(input3, 1);
        svint32x2_t input4 = svld2_s32(maskRest,(int*)(ptr1+4*nbValuesInVec));
        svint32_t input4_v0 = svget2_s32(input4, 0);
        svint32_t input4_v1 = svget2_s32(input4, 1);
        input4_v0 = svsel_s32(maskRest, input4_v0, intMaxVector);
        CoreSmallSort5(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0,
        input0_v1, input1_v1, input2_v1, input3_v1, input4_v1);
        svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
        svst2_s32(maskRest,(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
    break;
    }
    case 6 :
    {
        svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
        svint32_t input0_v0 = svget2_s32(input0, 0);
        svint32_t input0_v1 = svget2_s32(input0, 1);
        svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
        svint32_t input1_v0 = svget2_s32(input1, 0);
        svint32_t input1_v1 = svget2_s32(input1, 1);
        svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
        svint32_t input2_v0 = svget2_s32(input2, 0);
        svint32_t input2_v1 = svget2_s32(input2, 1);
        svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
        svint32_t input3_v0 = svget2_s32(input3, 0);
        svint32_t input3_v1 = svget2_s32(input3, 1);
        svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
        svint32_t input4_v0 = svget2_s32(input4, 0);
        svint32_t input4_v1 = svget2_s32(input4, 1);
        svint32x2_t input5 = svld2_s32(maskRest,(int*)(ptr1+5*nbValuesInVec));
        svint32_t input5_v0 = svget2_s32(input5, 0);
        svint32_t input5_v1 = svget2_s32(input5, 1);
        input5_v0 = svsel_s32(maskRest, input5_v0, intMaxVector);
        CoreSmallSort6(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0,
        input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1);
        svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
        svst2_s32(maskRest,(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));

    break;
    }
    case 7 :
    {
        svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
        svint32_t input0_v0 = svget2_s32(input0, 0);
        svint32_t input0_v1 = svget2_s32(input0, 1);
        svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
        svint32_t input1_v0 = svget2_s32(input1, 0);
        svint32_t input1_v1 = svget2_s32(input1, 1);
        svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
        svint32_t input2_v0 = svget2_s32(input2, 0);
        svint32_t input2_v1 = svget2_s32(input2, 1);
        svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
        svint32_t input3_v0 = svget2_s32(input3, 0);
        svint32_t input3_v1 = svget2_s32(input3, 1);
        svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
        svint32_t input4_v0 = svget2_s32(input4, 0);
        svint32_t input4_v1 = svget2_s32(input4, 1);
        svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
        svint32_t input5_v0 = svget2_s32(input5, 0);
        svint32_t input5_v1 = svget2_s32(input5, 1);
        svint32x2_t input6 = svld2_s32(maskRest,(int*)(ptr1+6*nbValuesInVec));
        svint32_t input6_v0 = svget2_s32(input6, 0);
        svint32_t input6_v1 = svget2_s32(input6, 1);
        input6_v0 = svsel_s32(maskRest, input6_v0, intMaxVector);
        CoreSmallSort7(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0,
        input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1);
        svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
        svst2_s32(maskRest,(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
        break;
    }
    case 8 :
    {
        svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
        svint32_t input0_v0 = svget2_s32(input0, 0);
        svint32_t input0_v1 = svget2_s32(input0, 1);
        svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
        svint32_t input1_v0 = svget2_s32(input1, 0);
        svint32_t input1_v1 = svget2_s32(input1, 1);
        svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
        svint32_t input2_v0 = svget2_s32(input2, 0);
        svint32_t input2_v1 = svget2_s32(input2, 1);
        svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
        svint32_t input3_v0 = svget2_s32(input3, 0);
        svint32_t input3_v1 = svget2_s32(input3, 1);
        svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
        svint32_t input4_v0 = svget2_s32(input4, 0);
        svint32_t input4_v1 = svget2_s32(input4, 1);
        svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
        svint32_t input5_v0 = svget2_s32(input5, 0);
        svint32_t input5_v1 = svget2_s32(input5, 1);
        svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
        svint32_t input6_v0 = svget2_s32(input6, 0);
        svint32_t input6_v1 = svget2_s32(input6, 1);
        svint32x2_t input7 = svld2_s32(maskRest,(int*)(ptr1+7*nbValuesInVec));
        svint32_t input7_v0 = svget2_s32(input7, 0);
        svint32_t input7_v1 = svget2_s32(input7, 1);
        input7_v0 = svsel_s32(maskRest, input7_v0, intMaxVector);
        CoreSmallSort8(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0,
        input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1);
        svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
        svst2_s32(maskRest,(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
    break;
    }
    case 9 :
    {
        svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
        svint32_t input0_v0 = svget2_s32(input0, 0);
        svint32_t input0_v1 = svget2_s32(input0, 1);
        svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
        svint32_t input1_v0 = svget2_s32(input1, 0);
        svint32_t input1_v1 = svget2_s32(input1, 1);
        svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
        svint32_t input2_v0 = svget2_s32(input2, 0);
        svint32_t input2_v1 = svget2_s32(input2, 1);
        svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
        svint32_t input3_v0 = svget2_s32(input3, 0);
        svint32_t input3_v1 = svget2_s32(input3, 1);
        svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
        svint32_t input4_v0 = svget2_s32(input4, 0);
        svint32_t input4_v1 = svget2_s32(input4, 1);
        svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
        svint32_t input5_v0 = svget2_s32(input5, 0);
        svint32_t input5_v1 = svget2_s32(input5, 1);
        svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
        svint32_t input6_v0 = svget2_s32(input6, 0);
        svint32_t input6_v1 = svget2_s32(input6, 1);
        svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
        svint32_t input7_v0 = svget2_s32(input7, 0);
        svint32_t input7_v1 = svget2_s32(input7, 1);
        svint32x2_t input8 = svld2_s32(maskRest,(int*)(ptr1+8*nbValuesInVec));
        svint32_t input8_v0 = svget2_s32(input8, 0);
        svint32_t input8_v1 = svget2_s32(input8, 1);
        input8_v0 = svsel_s32(maskRest, input8_v0, intMaxVector);
        CoreSmallSort9(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0, input8_v0,
        input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1, input8_v1);
        svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
        svst2_s32(maskRest,(int*)(ptr1+8*nbValuesInVec), svcreate2_s32(input8_v0,input8_v1));

    break;
    }
    case 10 :
    {
        svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
        svint32_t input0_v0 = svget2_s32(input0, 0);
        svint32_t input0_v1 = svget2_s32(input0, 1);
        svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
        svint32_t input1_v0 = svget2_s32(input1, 0);
        svint32_t input1_v1 = svget2_s32(input1, 1);
        svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
        svint32_t input2_v0 = svget2_s32(input2, 0);
        svint32_t input2_v1 = svget2_s32(input2, 1);
        svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
        svint32_t input3_v0 = svget2_s32(input3, 0);
        svint32_t input3_v1 = svget2_s32(input3, 1);
        svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
        svint32_t input4_v0 = svget2_s32(input4, 0);
        svint32_t input4_v1 = svget2_s32(input4, 1);
        svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
        svint32_t input5_v0 = svget2_s32(input5, 0);
        svint32_t input5_v1 = svget2_s32(input5, 1);
        svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
        svint32_t input6_v0 = svget2_s32(input6, 0);
        svint32_t input6_v1 = svget2_s32(input6, 1);
        svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
        svint32_t input7_v0 = svget2_s32(input7, 0);
        svint32_t input7_v1 = svget2_s32(input7, 1);
        svint32x2_t input8 = svld2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec));
        svint32_t input8_v0 = svget2_s32(input8, 0);
        svint32_t input8_v1 = svget2_s32(input8, 1);
        svint32x2_t input9 = svld2_s32(maskRest,(int*)(ptr1+9*nbValuesInVec));
        svint32_t input9_v0 = svget2_s32(input9, 0);
        svint32_t input9_v1 = svget2_s32(input9, 1);
        input9_v0 = svsel_s32(maskRest, input9_v0, intMaxVector);
        CoreSmallSort10(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0, input8_v0, input9_v0,
        input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1, input8_v1, input9_v1);
        svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec), svcreate2_s32(input8_v0,input8_v1));
        svst2_s32(maskRest,(int*)(ptr1+9*nbValuesInVec), svcreate2_s32(input9_v0,input9_v1));

    break;
    }
    case 11 :
    {
        svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
        svint32_t input0_v0 = svget2_s32(input0, 0);
        svint32_t input0_v1 = svget2_s32(input0, 1);
        svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
        svint32_t input1_v0 = svget2_s32(input1, 0);
        svint32_t input1_v1 = svget2_s32(input1, 1);
        svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
        svint32_t input2_v0 = svget2_s32(input2, 0);
        svint32_t input2_v1 = svget2_s32(input2, 1);
        svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
        svint32_t input3_v0 = svget2_s32(input3, 0);
        svint32_t input3_v1 = svget2_s32(input3, 1);
        svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
        svint32_t input4_v0 = svget2_s32(input4, 0);
        svint32_t input4_v1 = svget2_s32(input4, 1);
        svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
        svint32_t input5_v0 = svget2_s32(input5, 0);
        svint32_t input5_v1 = svget2_s32(input5, 1);
        svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
        svint32_t input6_v0 = svget2_s32(input6, 0);
        svint32_t input6_v1 = svget2_s32(input6, 1);
        svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
        svint32_t input7_v0 = svget2_s32(input7, 0);
        svint32_t input7_v1 = svget2_s32(input7, 1);
        svint32x2_t input8 = svld2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec));
        svint32_t input8_v0 = svget2_s32(input8, 0);
        svint32_t input8_v1 = svget2_s32(input8, 1);
        svint32x2_t input9 = svld2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec));
        svint32_t input9_v0 = svget2_s32(input9, 0);
        svint32_t input9_v1 = svget2_s32(input9, 1);
        svint32x2_t input10 = svld2_s32(maskRest,(int*)(ptr1+10*nbValuesInVec));
        svint32_t input10_v0 = svget2_s32(input10, 0);
        svint32_t input10_v1 = svget2_s32(input10, 1);
        input10_v0 = svsel_s32(maskRest, input10_v0, intMaxVector);
        CoreSmallSort11(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0, input8_v0, input9_v0, input10_v0,
        input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1, input8_v1, input9_v1, input10_v1);
        svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec), svcreate2_s32(input8_v0,input8_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec), svcreate2_s32(input9_v0,input9_v1));
        svst2_s32(maskRest,(int*)(ptr1+10*nbValuesInVec), svcreate2_s32(input10_v0,input10_v1));
    break;
    }
    case 12 :
    {
        svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
        svint32_t input0_v0 = svget2_s32(input0, 0);
        svint32_t input0_v1 = svget2_s32(input0, 1);
        svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
        svint32_t input1_v0 = svget2_s32(input1, 0);
        svint32_t input1_v1 = svget2_s32(input1, 1);
        svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
        svint32_t input2_v0 = svget2_s32(input2, 0);
        svint32_t input2_v1 = svget2_s32(input2, 1);
        svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
        svint32_t input3_v0 = svget2_s32(input3, 0);
        svint32_t input3_v1 = svget2_s32(input3, 1);
        svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
        svint32_t input4_v0 = svget2_s32(input4, 0);
        svint32_t input4_v1 = svget2_s32(input4, 1);
        svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
        svint32_t input5_v0 = svget2_s32(input5, 0);
        svint32_t input5_v1 = svget2_s32(input5, 1);
        svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
        svint32_t input6_v0 = svget2_s32(input6, 0);
        svint32_t input6_v1 = svget2_s32(input6, 1);
        svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
        svint32_t input7_v0 = svget2_s32(input7, 0);
        svint32_t input7_v1 = svget2_s32(input7, 1);
        svint32x2_t input8 = svld2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec));
        svint32_t input8_v0 = svget2_s32(input8, 0);
        svint32_t input8_v1 = svget2_s32(input8, 1);
        svint32x2_t input9 = svld2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec));
        svint32_t input9_v0 = svget2_s32(input9, 0);
        svint32_t input9_v1 = svget2_s32(input9, 1);
        svint32x2_t input10 = svld2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec));
        svint32_t input10_v0 = svget2_s32(input10, 0);
        svint32_t input10_v1 = svget2_s32(input10, 1);
        svint32x2_t input11 = svld2_s32(maskRest,(int*)(ptr1+11*nbValuesInVec));
        svint32_t input11_v0 = svget2_s32(input11, 0);
        svint32_t input11_v1 = svget2_s32(input11, 1);
        input11_v0 = svsel_s32(maskRest, input11_v0, intMaxVector);
        CoreSmallSort12(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0, input8_v0, input9_v0, input10_v0, input11_v0,
        input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1, input8_v1, input9_v1, input10_v1, input11_v1);
        svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec), svcreate2_s32(input8_v0,input8_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec), svcreate2_s32(input9_v0,input9_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec), svcreate2_s32(input10_v0,input10_v1));
        svst2_s32(maskRest,(int*)(ptr1+11*nbValuesInVec), svcreate2_s32(input11_v0,input11_v1));
    break;
    }
    case 13 :
    {
        svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
        svint32_t input0_v0 = svget2_s32(input0, 0);
        svint32_t input0_v1 = svget2_s32(input0, 1);
        svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
        svint32_t input1_v0 = svget2_s32(input1, 0);
        svint32_t input1_v1 = svget2_s32(input1, 1);
        svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
        svint32_t input2_v0 = svget2_s32(input2, 0);
        svint32_t input2_v1 = svget2_s32(input2, 1);
        svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
        svint32_t input3_v0 = svget2_s32(input3, 0);
        svint32_t input3_v1 = svget2_s32(input3, 1);
        svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
        svint32_t input4_v0 = svget2_s32(input4, 0);
        svint32_t input4_v1 = svget2_s32(input4, 1);
        svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
        svint32_t input5_v0 = svget2_s32(input5, 0);
        svint32_t input5_v1 = svget2_s32(input5, 1);
        svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
        svint32_t input6_v0 = svget2_s32(input6, 0);
        svint32_t input6_v1 = svget2_s32(input6, 1);
        svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
        svint32_t input7_v0 = svget2_s32(input7, 0);
        svint32_t input7_v1 = svget2_s32(input7, 1);
        svint32x2_t input8 = svld2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec));
        svint32_t input8_v0 = svget2_s32(input8, 0);
        svint32_t input8_v1 = svget2_s32(input8, 1);
        svint32x2_t input9 = svld2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec));
        svint32_t input9_v0 = svget2_s32(input9, 0);
        svint32_t input9_v1 = svget2_s32(input9, 1);
        svint32x2_t input10 = svld2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec));
        svint32_t input10_v0 = svget2_s32(input10, 0);
        svint32_t input10_v1 = svget2_s32(input10, 1);
        svint32x2_t input11 = svld2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec));
        svint32_t input11_v0 = svget2_s32(input11, 0);
        svint32_t input11_v1 = svget2_s32(input11, 1);
        svint32x2_t input12 = svld2_s32(maskRest,(int*)(ptr1+12*nbValuesInVec));
        svint32_t input12_v0 = svget2_s32(input12, 0);
        svint32_t input12_v1 = svget2_s32(input12, 1);
        input12_v0 = svsel_s32(maskRest, input12_v0, intMaxVector);
        CoreSmallSort13(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0, input8_v0, input9_v0, input10_v0, input11_v0, input12_v0,
        input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1, input8_v1, input9_v1, input10_v1, input11_v1, input12_v1);
        svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec), svcreate2_s32(input8_v0,input8_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec), svcreate2_s32(input9_v0,input9_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec), svcreate2_s32(input10_v0,input10_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec), svcreate2_s32(input11_v0,input11_v1));
        svst2_s32(maskRest,(int*)(ptr1+12*nbValuesInVec), svcreate2_s32(input12_v0,input12_v1));

    break;
    }
    case 14 :
    {
        svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
        svint32_t input0_v0 = svget2_s32(input0, 0);
        svint32_t input0_v1 = svget2_s32(input0, 1);
        svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
        svint32_t input1_v0 = svget2_s32(input1, 0);
        svint32_t input1_v1 = svget2_s32(input1, 1);
        svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
        svint32_t input2_v0 = svget2_s32(input2, 0);
        svint32_t input2_v1 = svget2_s32(input2, 1);
        svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
        svint32_t input3_v0 = svget2_s32(input3, 0);
        svint32_t input3_v1 = svget2_s32(input3, 1);
        svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
        svint32_t input4_v0 = svget2_s32(input4, 0);
        svint32_t input4_v1 = svget2_s32(input4, 1);
        svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
        svint32_t input5_v0 = svget2_s32(input5, 0);
        svint32_t input5_v1 = svget2_s32(input5, 1);
        svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
        svint32_t input6_v0 = svget2_s32(input6, 0);
        svint32_t input6_v1 = svget2_s32(input6, 1);
        svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
        svint32_t input7_v0 = svget2_s32(input7, 0);
        svint32_t input7_v1 = svget2_s32(input7, 1);
        svint32x2_t input8 = svld2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec));
        svint32_t input8_v0 = svget2_s32(input8, 0);
        svint32_t input8_v1 = svget2_s32(input8, 1);
        svint32x2_t input9 = svld2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec));
        svint32_t input9_v0 = svget2_s32(input9, 0);
        svint32_t input9_v1 = svget2_s32(input9, 1);
        svint32x2_t input10 = svld2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec));
        svint32_t input10_v0 = svget2_s32(input10, 0);
        svint32_t input10_v1 = svget2_s32(input10, 1);
        svint32x2_t input11 = svld2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec));
        svint32_t input11_v0 = svget2_s32(input11, 0);
        svint32_t input11_v1 = svget2_s32(input11, 1);
        svint32x2_t input12 = svld2_s32(svptrue_b32(),(int*)(ptr1+12*nbValuesInVec));
        svint32_t input12_v0 = svget2_s32(input12, 0);
        svint32_t input12_v1 = svget2_s32(input12, 1);
        svint32x2_t input13 = svld2_s32(maskRest,(int*)(ptr1+13*nbValuesInVec));
        svint32_t input13_v0 = svget2_s32(input13, 0);
        svint32_t input13_v1 = svget2_s32(input13, 1);
        input13_v0 = svsel_s32(maskRest, input13_v0, intMaxVector);
        CoreSmallSort14(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0, input8_v0, input9_v0, input10_v0, input11_v0, input12_v0, input13_v0,
        input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1, input8_v1, input9_v1, input10_v1, input11_v1, input12_v1, input13_v1);
        svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec), svcreate2_s32(input8_v0,input8_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec), svcreate2_s32(input9_v0,input9_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec), svcreate2_s32(input10_v0,input10_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec), svcreate2_s32(input11_v0,input11_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+12*nbValuesInVec), svcreate2_s32(input12_v0,input12_v1));
        svst2_s32(maskRest,(int*)(ptr1+13*nbValuesInVec), svcreate2_s32(input13_v0,input13_v1));

    break;
    }
    case 15 :
    {
        svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
        svint32_t input0_v0 = svget2_s32(input0, 0);
        svint32_t input0_v1 = svget2_s32(input0, 1);
        svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
        svint32_t input1_v0 = svget2_s32(input1, 0);
        svint32_t input1_v1 = svget2_s32(input1, 1);
        svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
        svint32_t input2_v0 = svget2_s32(input2, 0);
        svint32_t input2_v1 = svget2_s32(input2, 1);
        svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
        svint32_t input3_v0 = svget2_s32(input3, 0);
        svint32_t input3_v1 = svget2_s32(input3, 1);
        svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
        svint32_t input4_v0 = svget2_s32(input4, 0);
        svint32_t input4_v1 = svget2_s32(input4, 1);
        svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
        svint32_t input5_v0 = svget2_s32(input5, 0);
        svint32_t input5_v1 = svget2_s32(input5, 1);
        svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
        svint32_t input6_v0 = svget2_s32(input6, 0);
        svint32_t input6_v1 = svget2_s32(input6, 1);
        svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
        svint32_t input7_v0 = svget2_s32(input7, 0);
        svint32_t input7_v1 = svget2_s32(input7, 1);
        svint32x2_t input8 = svld2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec));
        svint32_t input8_v0 = svget2_s32(input8, 0);
        svint32_t input8_v1 = svget2_s32(input8, 1);
        svint32x2_t input9 = svld2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec));
        svint32_t input9_v0 = svget2_s32(input9, 0);
        svint32_t input9_v1 = svget2_s32(input9, 1);
        svint32x2_t input10 = svld2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec));
        svint32_t input10_v0 = svget2_s32(input10, 0);
        svint32_t input10_v1 = svget2_s32(input10, 1);
        svint32x2_t input11 = svld2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec));
        svint32_t input11_v0 = svget2_s32(input11, 0);
        svint32_t input11_v1 = svget2_s32(input11, 1);
        svint32x2_t input12 = svld2_s32(svptrue_b32(),(int*)(ptr1+12*nbValuesInVec));
        svint32_t input12_v0 = svget2_s32(input12, 0);
        svint32_t input12_v1 = svget2_s32(input12, 1);
        svint32x2_t input13 = svld2_s32(svptrue_b32(),(int*)(ptr1+13*nbValuesInVec));
        svint32_t input13_v0 = svget2_s32(input13, 0);
        svint32_t input13_v1 = svget2_s32(input13, 1);
        svint32x2_t input14 = svld2_s32(maskRest,(int*)(ptr1+14*nbValuesInVec));
        svint32_t input14_v0 = svget2_s32(input14, 0);
        svint32_t input14_v1 = svget2_s32(input14, 1);
        input14_v0 = svsel_s32(maskRest, input14_v0, intMaxVector);
        CoreSmallSort15(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0, input8_v0, input9_v0, input10_v0, input11_v0, input12_v0, input13_v0, input14_v0,
        input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1, input8_v1, input9_v1, input10_v1, input11_v1, input12_v1, input13_v1, input14_v1);
        svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec), svcreate2_s32(input8_v0,input8_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec), svcreate2_s32(input9_v0,input9_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec), svcreate2_s32(input10_v0,input10_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec), svcreate2_s32(input11_v0,input11_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+12*nbValuesInVec), svcreate2_s32(input12_v0,input12_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+13*nbValuesInVec), svcreate2_s32(input13_v0,input13_v1));
        svst2_s32(maskRest,(int*)(ptr1+14*nbValuesInVec), svcreate2_s32(input14_v0,input14_v1));

    break;
    }
    case 16 :
    {
        svint32x2_t input0 = svld2_s32(svptrue_b32(),(int*)ptr1);
        svint32_t input0_v0 = svget2_s32(input0, 0);
        svint32_t input0_v1 = svget2_s32(input0, 1);
        svint32x2_t input1 = svld2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec));
        svint32_t input1_v0 = svget2_s32(input1, 0);
        svint32_t input1_v1 = svget2_s32(input1, 1);
        svint32x2_t input2 = svld2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec));
        svint32_t input2_v0 = svget2_s32(input2, 0);
        svint32_t input2_v1 = svget2_s32(input2, 1);
        svint32x2_t input3 = svld2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec));
        svint32_t input3_v0 = svget2_s32(input3, 0);
        svint32_t input3_v1 = svget2_s32(input3, 1);
        svint32x2_t input4 = svld2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec));
        svint32_t input4_v0 = svget2_s32(input4, 0);
        svint32_t input4_v1 = svget2_s32(input4, 1);
        svint32x2_t input5 = svld2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec));
        svint32_t input5_v0 = svget2_s32(input5, 0);
        svint32_t input5_v1 = svget2_s32(input5, 1);
        svint32x2_t input6 = svld2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec));
        svint32_t input6_v0 = svget2_s32(input6, 0);
        svint32_t input6_v1 = svget2_s32(input6, 1);
        svint32x2_t input7 = svld2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec));
        svint32_t input7_v0 = svget2_s32(input7, 0);
        svint32_t input7_v1 = svget2_s32(input7, 1);
        svint32x2_t input8 = svld2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec));
        svint32_t input8_v0 = svget2_s32(input8, 0);
        svint32_t input8_v1 = svget2_s32(input8, 1);
        svint32x2_t input9 = svld2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec));
        svint32_t input9_v0 = svget2_s32(input9, 0);
        svint32_t input9_v1 = svget2_s32(input9, 1);
        svint32x2_t input10 = svld2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec));
        svint32_t input10_v0 = svget2_s32(input10, 0);
        svint32_t input10_v1 = svget2_s32(input10, 1);
        svint32x2_t input11 = svld2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec));
        svint32_t input11_v0 = svget2_s32(input11, 0);
        svint32_t input11_v1 = svget2_s32(input11, 1);
        svint32x2_t input12 = svld2_s32(svptrue_b32(),(int*)(ptr1+12*nbValuesInVec));
        svint32_t input12_v0 = svget2_s32(input12, 0);
        svint32_t input12_v1 = svget2_s32(input12, 1);
        svint32x2_t input13 = svld2_s32(svptrue_b32(),(int*)(ptr1+13*nbValuesInVec));
        svint32_t input13_v0 = svget2_s32(input13, 0);
        svint32_t input13_v1 = svget2_s32(input13, 1);
        svint32x2_t input14 = svld2_s32(svptrue_b32(),(int*)(ptr1+14*nbValuesInVec));
        svint32_t input14_v0 = svget2_s32(input14, 0);
        svint32_t input14_v1 = svget2_s32(input14, 1);
        svint32x2_t input15 = svld2_s32(maskRest,(int*)(ptr1+15*nbValuesInVec));
        svint32_t input15_v0 = svget2_s32(input15, 0);
        svint32_t input15_v1 = svget2_s32(input15, 1);
        input15_v0 = svsel_s32(maskRest, input15_v0, intMaxVector);
        CoreSmallSort16(input0_v0, input1_v0, input2_v0, input3_v0, input4_v0, input5_v0, input6_v0, input7_v0, input8_v0, input9_v0, input10_v0, input11_v0, input12_v0, input13_v0, input14_v0, input15_v0,
        input0_v1, input1_v1, input2_v1, input3_v1, input4_v1, input5_v1, input6_v1, input7_v1, input8_v1, input9_v1, input10_v1, input11_v1, input12_v1, input13_v1, input14_v1, input15_v1);
        svst2_s32(svptrue_b32(),(int*)ptr1, svcreate2_s32(input0_v0,input0_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+nbValuesInVec), svcreate2_s32(input1_v0,input1_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+2*nbValuesInVec), svcreate2_s32(input2_v0,input2_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+3*nbValuesInVec), svcreate2_s32(input3_v0,input3_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+4*nbValuesInVec), svcreate2_s32(input4_v0,input4_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+5*nbValuesInVec), svcreate2_s32(input5_v0,input5_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+6*nbValuesInVec), svcreate2_s32(input6_v0,input6_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+7*nbValuesInVec), svcreate2_s32(input7_v0,input7_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+8*nbValuesInVec), svcreate2_s32(input8_v0,input8_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+9*nbValuesInVec), svcreate2_s32(input9_v0,input9_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+10*nbValuesInVec), svcreate2_s32(input10_v0,input10_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+11*nbValuesInVec), svcreate2_s32(input11_v0,input11_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+12*nbValuesInVec), svcreate2_s32(input12_v0,input12_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+13*nbValuesInVec), svcreate2_s32(input13_v0,input13_v1));
        svst2_s32(svptrue_b32(),(int*)(ptr1+14*nbValuesInVec), svcreate2_s32(input14_v0,input14_v1));
        svst2_s32(maskRest,(int*)(ptr1+15*nbValuesInVec), svcreate2_s32(input15_v0,input15_v1));
    break;
    }
    }
}

inline void SmallSort16V(int* __restrict__ ptr, int* __restrict__ values, const size_t length){
    // length is limited to 4 times size of a vec
    const int nbValuesInVec = svcntw();
    const int nbVecs = (length+nbValuesInVec-1)/nbValuesInVec;
    const int rest = nbVecs*nbValuesInVec-length;
    const int lastVecSize = nbValuesInVec-rest;

    const svint32_t intMaxVector = svdup_s32(INT_MAX);
    const svbool_t maskRest = SortSVE::getTrueFalseMask32(lastVecSize);

    switch(nbVecs){
    case 1:
    {
        svint32_t v1 = svsel_s32(maskRest, svld1_s32(maskRest,ptr), intMaxVector);
        svint32_t v1_val = svsel_s32(maskRest, svld1_s32(maskRest,values), intMaxVector);
        CoreSmallSort(v1,
                           v1_val);
        svst1_s32(maskRest, ptr, v1);
        svst1_s32(maskRest, values, v1_val);
    }
        break;
    case 2:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v1_val = svld1_s32(svptrue_b32(),values);
        svint32_t v2 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec), intMaxVector);
        svint32_t v2_val = svsel_s32(maskRest, svld1_s32(maskRest,values+nbValuesInVec), intMaxVector);
        CoreSmallSort2(v1,v2,
                       v1_val,v2_val);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), values, v1_val);
        svst1_s32(maskRest, ptr+nbValuesInVec, v2);
        svst1_s32(maskRest, values+nbValuesInVec, v2_val);
    }
        break;
    case 3:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v1_val = svld1_s32(svptrue_b32(),values);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v2_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
        svint32_t v3 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*2), intMaxVector);
        svint32_t v3_val = svsel_s32(maskRest, svld1_s32(maskRest,values+nbValuesInVec*2), intMaxVector);
        CoreSmallSort3(v1,v2,v3,
                       v1_val,v2_val,v3_val);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), values, v1_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), values+nbValuesInVec, v2_val);
        svst1_s32(maskRest, ptr+nbValuesInVec*2, v3);
        svst1_s32(maskRest, values+nbValuesInVec*2, v3_val);
    }
        break;
    case 4:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v1_val = svld1_s32(svptrue_b32(),values);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v2_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v3_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
        svint32_t v4 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*3), intMaxVector);
        svint32_t v4_val = svsel_s32(maskRest, svld1_s32(maskRest,values+nbValuesInVec*3), intMaxVector);
        CoreSmallSort4(v1,v2,v3,v4,
                       v1_val,v2_val,v3_val,v4_val);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), values, v1_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), values+nbValuesInVec, v2_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*2, v3_val);
        svst1_s32(maskRest, ptr+nbValuesInVec*3, v4);
        svst1_s32(maskRest, values+nbValuesInVec*3, v4_val);
    }
        break;
    case 5:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v1_val = svld1_s32(svptrue_b32(),values);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v2_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v3_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v4_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
        svint32_t v5 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*4), intMaxVector);
        svint32_t v5_val = svsel_s32(maskRest, svld1_s32(maskRest,values+nbValuesInVec*4), intMaxVector);
        CoreSmallSort5(v1,v2,v3,v4,v5,
                       v1_val,v2_val,v3_val,v4_val,v5_val);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), values, v1_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), values+nbValuesInVec, v2_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*2, v3_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*3, v4_val);
        svst1_s32(maskRest, ptr+nbValuesInVec*4, v5);
        svst1_s32(maskRest, values+nbValuesInVec*4, v5_val);
    }
        break;
    case 6:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v1_val = svld1_s32(svptrue_b32(),values);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v2_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v3_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v4_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v5_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
        svint32_t v6 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*5), intMaxVector);
        svint32_t v6_val = svsel_s32(maskRest, svld1_s32(maskRest,values+nbValuesInVec*5), intMaxVector);
        CoreSmallSort6(v1,v2,v3,v4,v5, v6,
                       v1_val,v2_val,v3_val,v4_val,v5_val,v6_val);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), values, v1_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), values+nbValuesInVec, v2_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*2, v3_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*3, v4_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*4, v5_val);
        svst1_s32(maskRest, ptr+nbValuesInVec*5, v6);
        svst1_s32(maskRest, values+nbValuesInVec*5, v6_val);
    }
        break;
    case 7:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v1_val = svld1_s32(svptrue_b32(),values);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v2_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v3_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v4_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v5_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v6_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
        svint32_t v7 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*6), intMaxVector);
        svint32_t v7_val = svsel_s32(maskRest, svld1_s32(maskRest,values+nbValuesInVec*6), intMaxVector);
        CoreSmallSort7(v1,v2,v3,v4,v5,v6,v7,
                       v1_val,v2_val,v3_val,v4_val,v5_val,v6_val,v7_val);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), values, v1_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), values+nbValuesInVec, v2_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*2, v3_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*3, v4_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*4, v5_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*5, v6_val);
        svst1_s32(maskRest, ptr+nbValuesInVec*6, v7);
        svst1_s32(maskRest, values+nbValuesInVec*6, v7_val);
    }
        break;
    case 8:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v1_val = svld1_s32(svptrue_b32(),values);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v2_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v3_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v4_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v5_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v6_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v7_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
        svint32_t v8 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*7), intMaxVector);
        svint32_t v8_val = svsel_s32(maskRest, svld1_s32(maskRest,values+nbValuesInVec*7), intMaxVector);
        CoreSmallSort8(v1,v2,v3,v4,v5,v6,v7,v8,
                       v1_val,v2_val,v3_val,v4_val,v5_val,v6_val,v7_val,v8_val);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), values, v1_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), values+nbValuesInVec, v2_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*2, v3_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*3, v4_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*4, v5_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*5, v6_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*6, v7_val);
        svst1_s32(maskRest, ptr+nbValuesInVec*7, v8);
        svst1_s32(maskRest, values+nbValuesInVec*7, v8_val);
    }
        break;
    case 9:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v1_val = svld1_s32(svptrue_b32(),values);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v2_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v3_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v4_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v5_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v6_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v7_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
        svint32_t v8 = svld1_s32(svptrue_b32(),ptr+7*nbValuesInVec);
        svint32_t v8_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
        svint32_t v9 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*8), intMaxVector);
        svint32_t v9_val = svsel_s32(maskRest, svld1_s32(maskRest,values+nbValuesInVec*8), intMaxVector);
        CoreSmallSort9(v1,v2,v3,v4,v5,v6,v7,v8,v9,
                       v1_val,v2_val,v3_val,v4_val,v5_val,v6_val,v7_val,v8_val,v9_val);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), values, v1_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), values+nbValuesInVec, v2_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*2, v3_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*3, v4_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*4, v5_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*5, v6_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*6, v7_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*7, v8);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*7, v8_val);
        svst1_s32(maskRest, ptr+nbValuesInVec*8, v9);
        svst1_s32(maskRest, values+nbValuesInVec*8, v9_val);
    }
        break;
    case 10:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v1_val = svld1_s32(svptrue_b32(),values);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v2_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v3_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v4_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v5_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v6_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v7_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
        svint32_t v8 = svld1_s32(svptrue_b32(),ptr+7*nbValuesInVec);
        svint32_t v8_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
        svint32_t v9 = svld1_s32(svptrue_b32(),ptr+8*nbValuesInVec);
        svint32_t v9_val = svld1_s32(svptrue_b32(),values+8*nbValuesInVec);
        svint32_t v10 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*9), intMaxVector);
        svint32_t v10_val = svsel_s32(maskRest, svld1_s32(maskRest,values+nbValuesInVec*9), intMaxVector);
        CoreSmallSort10(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
                        v1_val,v2_val,v3_val,v4_val,v5_val,v6_val,v7_val,v8_val,v9_val,v10_val);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), values, v1_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), values+nbValuesInVec, v2_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*2, v3_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*3, v4_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*4, v5_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*5, v6_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*6, v7_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*7, v8);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*7, v8_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*8, v9);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*8, v9_val);
        svst1_s32(maskRest, ptr+nbValuesInVec*9, v10);
        svst1_s32(maskRest, values+nbValuesInVec*9, v10_val);
    }
        break;
    case 11:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v1_val = svld1_s32(svptrue_b32(),values);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v2_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v3_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v4_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v5_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v6_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v7_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
        svint32_t v8 = svld1_s32(svptrue_b32(),ptr+7*nbValuesInVec);
        svint32_t v8_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
        svint32_t v9 = svld1_s32(svptrue_b32(),ptr+8*nbValuesInVec);
        svint32_t v9_val = svld1_s32(svptrue_b32(),values+8*nbValuesInVec);
        svint32_t v10 = svld1_s32(svptrue_b32(),ptr+9*nbValuesInVec);
        svint32_t v10_val = svld1_s32(svptrue_b32(),values+9*nbValuesInVec);
        svint32_t v11 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*10), intMaxVector);
        svint32_t v11_val = svsel_s32(maskRest, svld1_s32(maskRest,values+nbValuesInVec*10), intMaxVector);
        CoreSmallSort11(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,
                        v1_val,v2_val,v3_val,v4_val,v5_val,v6_val,v7_val,v8_val,v9_val,v10_val,v11_val);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), values, v1_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), values+nbValuesInVec, v2_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*2, v3_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*3, v4_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*4, v5_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*5, v6_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*6, v7_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*7, v8);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*7, v8_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*8, v9);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*8, v9_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*9, v10);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*9, v10_val);
        svst1_s32(maskRest, ptr+nbValuesInVec*10, v11);
        svst1_s32(maskRest, values+nbValuesInVec*10, v11_val);
    }
        break;
    case 12:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v1_val = svld1_s32(svptrue_b32(),values);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v2_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v3_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v4_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v5_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v6_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v7_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
        svint32_t v8 = svld1_s32(svptrue_b32(),ptr+7*nbValuesInVec);
        svint32_t v8_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
        svint32_t v9 = svld1_s32(svptrue_b32(),ptr+8*nbValuesInVec);
        svint32_t v9_val = svld1_s32(svptrue_b32(),values+8*nbValuesInVec);
        svint32_t v10 = svld1_s32(svptrue_b32(),ptr+9*nbValuesInVec);
        svint32_t v10_val = svld1_s32(svptrue_b32(),values+9*nbValuesInVec);
        svint32_t v11 = svld1_s32(svptrue_b32(),ptr+10*nbValuesInVec);
        svint32_t v11_val = svld1_s32(svptrue_b32(),values+10*nbValuesInVec);
        svint32_t v12 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*11), intMaxVector);
        svint32_t v12_val = svsel_s32(maskRest, svld1_s32(maskRest,values+nbValuesInVec*11), intMaxVector);
        CoreSmallSort12(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,
                        v1_val,v2_val,v3_val,v4_val,v5_val,v6_val,v7_val,v8_val,v9_val,v10_val,v11_val,v12_val);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), values, v1_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), values+nbValuesInVec, v2_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*2, v3_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*3, v4_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*4, v5_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*5, v6_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*6, v7_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*7, v8);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*7, v8_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*8, v9);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*8, v9_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*9, v10);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*9, v10_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*10, v11);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*10, v11_val);
        svst1_s32(maskRest, ptr+nbValuesInVec*11, v12);
        svst1_s32(maskRest, values+nbValuesInVec*11, v12_val);
    }
        break;
    case 13:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v1_val = svld1_s32(svptrue_b32(),values);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v2_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v3_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v4_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v5_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v6_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v7_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
        svint32_t v8 = svld1_s32(svptrue_b32(),ptr+7*nbValuesInVec);
        svint32_t v8_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
        svint32_t v9 = svld1_s32(svptrue_b32(),ptr+8*nbValuesInVec);
        svint32_t v9_val = svld1_s32(svptrue_b32(),values+8*nbValuesInVec);
        svint32_t v10 = svld1_s32(svptrue_b32(),ptr+9*nbValuesInVec);
        svint32_t v10_val = svld1_s32(svptrue_b32(),values+9*nbValuesInVec);
        svint32_t v11 = svld1_s32(svptrue_b32(),ptr+10*nbValuesInVec);
        svint32_t v11_val = svld1_s32(svptrue_b32(),values+10*nbValuesInVec);
        svint32_t v12 = svld1_s32(svptrue_b32(),ptr+11*nbValuesInVec);
        svint32_t v12_val = svld1_s32(svptrue_b32(),values+11*nbValuesInVec);
        svint32_t v13 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*12), intMaxVector);
        svint32_t v13_val = svsel_s32(maskRest, svld1_s32(maskRest,values+nbValuesInVec*12), intMaxVector);
        CoreSmallSort13(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,
                        v1_val,v2_val,v3_val,v4_val,v5_val,v6_val,v7_val,v8_val,v9_val,v10_val,v11_val,v12_val,v13_val);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), values, v1_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), values+nbValuesInVec, v2_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*2, v3_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*3, v4_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*4, v5_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*5, v6_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*6, v7_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*7, v8);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*7, v8_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*8, v9);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*8, v9_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*9, v10);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*9, v10_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*10, v11);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*10, v11_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*11, v12);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*11, v12_val);
        svst1_s32(maskRest, ptr+nbValuesInVec*12, v13);
        svst1_s32(maskRest, values+nbValuesInVec*12, v13_val);
    }
        break;
    case 14:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v1_val = svld1_s32(svptrue_b32(),values);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v2_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v3_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v4_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v5_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v6_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v7_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
        svint32_t v8 = svld1_s32(svptrue_b32(),ptr+7*nbValuesInVec);
        svint32_t v8_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
        svint32_t v9 = svld1_s32(svptrue_b32(),ptr+8*nbValuesInVec);
        svint32_t v9_val = svld1_s32(svptrue_b32(),values+8*nbValuesInVec);
        svint32_t v10 = svld1_s32(svptrue_b32(),ptr+9*nbValuesInVec);
        svint32_t v10_val = svld1_s32(svptrue_b32(),values+9*nbValuesInVec);
        svint32_t v11 = svld1_s32(svptrue_b32(),ptr+10*nbValuesInVec);
        svint32_t v11_val = svld1_s32(svptrue_b32(),values+10*nbValuesInVec);
        svint32_t v12 = svld1_s32(svptrue_b32(),ptr+11*nbValuesInVec);
        svint32_t v12_val = svld1_s32(svptrue_b32(),values+11*nbValuesInVec);
        svint32_t v13 = svld1_s32(svptrue_b32(),ptr+12*nbValuesInVec);
        svint32_t v13_val = svld1_s32(svptrue_b32(),values+12*nbValuesInVec);
        svint32_t v14 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*13), intMaxVector);
        svint32_t v14_val = svsel_s32(maskRest, svld1_s32(maskRest,values+nbValuesInVec*13), intMaxVector);
        CoreSmallSort14(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,
                        v1_val,v2_val,v3_val,v4_val,v5_val,v6_val,v7_val,v8_val,v9_val,v10_val,v11_val,v12_val,v13_val,v14_val);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), values, v1_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), values+nbValuesInVec, v2_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*2, v3_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*3, v4_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*4, v5_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*5, v6_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*6, v7_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*7, v8);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*7, v8_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*8, v9);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*8, v9_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*9, v10);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*9, v10_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*10, v11);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*10, v11_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*11, v12);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*11, v12_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*12, v13);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*12, v13_val);
        svst1_s32(maskRest, ptr+nbValuesInVec*13, v14);
        svst1_s32(maskRest, values+nbValuesInVec*13, v14_val);
    }
        break;
    case 15:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v1_val = svld1_s32(svptrue_b32(),values);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v2_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v3_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v4_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v5_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v6_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v7_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
        svint32_t v8 = svld1_s32(svptrue_b32(),ptr+7*nbValuesInVec);
        svint32_t v8_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
        svint32_t v9 = svld1_s32(svptrue_b32(),ptr+8*nbValuesInVec);
        svint32_t v9_val = svld1_s32(svptrue_b32(),values+8*nbValuesInVec);
        svint32_t v10 = svld1_s32(svptrue_b32(),ptr+9*nbValuesInVec);
        svint32_t v10_val = svld1_s32(svptrue_b32(),values+9*nbValuesInVec);
        svint32_t v11 = svld1_s32(svptrue_b32(),ptr+10*nbValuesInVec);
        svint32_t v11_val = svld1_s32(svptrue_b32(),values+10*nbValuesInVec);
        svint32_t v12 = svld1_s32(svptrue_b32(),ptr+11*nbValuesInVec);
        svint32_t v12_val = svld1_s32(svptrue_b32(),values+11*nbValuesInVec);
        svint32_t v13 = svld1_s32(svptrue_b32(),ptr+12*nbValuesInVec);
        svint32_t v13_val = svld1_s32(svptrue_b32(),values+12*nbValuesInVec);
        svint32_t v14 = svld1_s32(svptrue_b32(),ptr+13*nbValuesInVec);
        svint32_t v14_val = svld1_s32(svptrue_b32(),values+13*nbValuesInVec);
        svint32_t v15 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*14), intMaxVector);
        svint32_t v15_val = svsel_s32(maskRest, svld1_s32(maskRest,values+nbValuesInVec*14), intMaxVector);
        CoreSmallSort15(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,
                        v1_val,v2_val,v3_val,v4_val,v5_val,v6_val,v7_val,v8_val,v9_val,v10_val,v11_val,v12_val,v13_val,v14_val,v15_val);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), values, v1_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), values+nbValuesInVec, v2_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*2, v3_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*3, v4_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*4, v5_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*5, v6_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*6, v7_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*7, v8);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*7, v8_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*8, v9);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*8, v9_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*9, v10);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*9, v10_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*10, v11);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*10, v11_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*11, v12);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*11, v12_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*12, v13);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*12, v13_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*13, v14);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*13, v14_val);
        svst1_s32(maskRest, ptr+nbValuesInVec*14, v15);
        svst1_s32(maskRest, values+nbValuesInVec*14, v15_val);
    }
        break;
        //case 16:
    default:
    {
        svint32_t v1 = svld1_s32(svptrue_b32(),ptr);
        svint32_t v1_val = svld1_s32(svptrue_b32(),values);
        svint32_t v2 = svld1_s32(svptrue_b32(),ptr+nbValuesInVec);
        svint32_t v2_val = svld1_s32(svptrue_b32(),values+nbValuesInVec);
        svint32_t v3 = svld1_s32(svptrue_b32(),ptr+2*nbValuesInVec);
        svint32_t v3_val = svld1_s32(svptrue_b32(),values+2*nbValuesInVec);
        svint32_t v4 = svld1_s32(svptrue_b32(),ptr+3*nbValuesInVec);
        svint32_t v4_val = svld1_s32(svptrue_b32(),values+3*nbValuesInVec);
        svint32_t v5 = svld1_s32(svptrue_b32(),ptr+4*nbValuesInVec);
        svint32_t v5_val = svld1_s32(svptrue_b32(),values+4*nbValuesInVec);
        svint32_t v6 = svld1_s32(svptrue_b32(),ptr+5*nbValuesInVec);
        svint32_t v6_val = svld1_s32(svptrue_b32(),values+5*nbValuesInVec);
        svint32_t v7 = svld1_s32(svptrue_b32(),ptr+6*nbValuesInVec);
        svint32_t v7_val = svld1_s32(svptrue_b32(),values+6*nbValuesInVec);
        svint32_t v8 = svld1_s32(svptrue_b32(),ptr+7*nbValuesInVec);
        svint32_t v8_val = svld1_s32(svptrue_b32(),values+7*nbValuesInVec);
        svint32_t v9 = svld1_s32(svptrue_b32(),ptr+8*nbValuesInVec);
        svint32_t v9_val = svld1_s32(svptrue_b32(),values+8*nbValuesInVec);
        svint32_t v10 = svld1_s32(svptrue_b32(),ptr+9*nbValuesInVec);
        svint32_t v10_val = svld1_s32(svptrue_b32(),values+9*nbValuesInVec);
        svint32_t v11 = svld1_s32(svptrue_b32(),ptr+10*nbValuesInVec);
        svint32_t v11_val = svld1_s32(svptrue_b32(),values+10*nbValuesInVec);
        svint32_t v12 = svld1_s32(svptrue_b32(),ptr+11*nbValuesInVec);
        svint32_t v12_val = svld1_s32(svptrue_b32(),values+11*nbValuesInVec);
        svint32_t v13 = svld1_s32(svptrue_b32(),ptr+12*nbValuesInVec);
        svint32_t v13_val = svld1_s32(svptrue_b32(),values+12*nbValuesInVec);
        svint32_t v14 = svld1_s32(svptrue_b32(),ptr+13*nbValuesInVec);
        svint32_t v14_val = svld1_s32(svptrue_b32(),values+13*nbValuesInVec);
        svint32_t v15 = svld1_s32(svptrue_b32(),ptr+14*nbValuesInVec);
        svint32_t v15_val = svld1_s32(svptrue_b32(),values+14*nbValuesInVec);
        svint32_t v16 = svsel_s32(maskRest, svld1_s32(maskRest,ptr+nbValuesInVec*15), intMaxVector);
        svint32_t v16_val = svsel_s32(maskRest, svld1_s32(maskRest,values+nbValuesInVec*15), intMaxVector);
        CoreSmallSort16(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,
                        v1_val,v2_val,v3_val,v4_val,v5_val,v6_val,v7_val,v8_val,v9_val,v10_val,v11_val,v12_val,v13_val,v14_val,v15_val,v16_val);
        svst1_s32(svptrue_b32(), ptr, v1);
        svst1_s32(svptrue_b32(), values, v1_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec, v2);
        svst1_s32(svptrue_b32(), values+nbValuesInVec, v2_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*2, v3);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*2, v3_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*3, v4);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*3, v4_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*4, v5);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*4, v5_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*5, v6);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*5, v6_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*6, v7);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*6, v7_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*7, v8);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*7, v8_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*8, v9);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*8, v9_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*9, v10);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*9, v10_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*10, v11);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*10, v11_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*11, v12);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*11, v12_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*12, v13);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*12, v13_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*13, v14);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*13, v14_val);
        svst1_s32(svptrue_b32(), ptr+nbValuesInVec*14, v15);
        svst1_s32(svptrue_b32(), values+nbValuesInVec*14, v15_val);
        svst1_s32(maskRest, ptr+nbValuesInVec*15, v16);
        svst1_s32(maskRest, values+nbValuesInVec*15, v16_val);
    }
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Partitions
////////////////////////////////////////////////////////////////////////////////

template <class SortType, class IndexType>
static inline IndexType CoreScalarPartition(SortType array[], SortType values[], IndexType left, IndexType right,
                                            const SortType pivot){

    for(; left <= right
        && array[left] <= pivot ; ++left){
    }

    for(IndexType idx = left ; idx <= right ; ++idx){
        if( array[idx] <= pivot ){
            std::swap(array[idx],array[left]);
            std::swap(values[idx],values[left]);
            left += 1;
        }
    }

    return left;
}

template <class SortType, class IndexType>
static inline IndexType CoreScalarPartition(std::pair<SortType,SortType> array[], IndexType left, IndexType right,
                                            const SortType pivot){

    for(; left <= right
        && array[left].first <= pivot ; ++left){
    }

    for(IndexType idx = left ; idx <= right ; ++idx){
        if( array[idx].first <= pivot ){
            std::swap(array[idx],array[left]);
            left += 1;
        }
    }

    return left;
}

/* a sequential qs */
template <class IndexType>
static inline IndexType PartitionSVE(int array[], int values[], IndexType left, IndexType right,
                                     const int pivot){
    const IndexType S = svcntw();

    if(right-left+1 < 2*S){
        return CoreScalarPartition<int,IndexType>(array, values, left, right, pivot);
    }

    svint32_t pivotvec = svdup_s32(pivot);

    svint32_t left_val = svld1_s32(svptrue_b32(),&array[left]);
    svint32_t left_val_val = svld1_s32(svptrue_b32(),&values[left]);
    IndexType left_w = left;
    left += S;

    IndexType right_w = right+1;
    right -= S-1;
    svint32_t right_val = svld1_s32(svptrue_b32(),&array[right]);
    svint32_t right_val_val = svld1_s32(svptrue_b32(),&values[right]);

    while(left + S <= right){
        const IndexType free_left = left - left_w;
        const IndexType free_right = right_w - right;

        svint32_t val;
        svint32_t val_val;
        if( free_left <= free_right ){
            val = svld1_s32(svptrue_b32(),&array[left]);
            val_val = svld1_s32(svptrue_b32(),&values[left]);
            left += S;
        }
        else{
            right -= S;
            val = svld1_s32(svptrue_b32(),&array[right]);
            val_val = svld1_s32(svptrue_b32(),&values[right]);
        }

        svbool_t mask = svcmple_s32(svptrue_b32(), val, pivotvec);

        const IndexType nb_low = svcntp_b32(svptrue_b32(),mask);
        const IndexType nb_high = S-nb_low;

        svint32_t val_low = svcompact_s32(mask, val);
        svint32_t val_val_low = svcompact_s32(mask, val_val);
        svbool_t mask_comp_high = svnot_b_z(svptrue_b32(), mask);
        svint32_t val_high = svcompact_s32(mask_comp_high, val);
        svint32_t val_val_high = svcompact_s32(mask_comp_high, val_val);

        svbool_t mask_low = SortSVE::getTrueFalseMask32(nb_low);
        svst1_s32(mask_low,&array[left_w],val_low);
        svst1_s32(mask_low,&values[left_w],val_val_low);
        left_w += nb_low;

        right_w -= nb_high;
        svbool_t mask_high = SortSVE::getTrueFalseMask32(nb_high);
        svst1_s32(mask_high,&array[right_w],val_high);
        svst1_s32(mask_high,&values[right_w],val_val_high);
    }

    {
        const IndexType remaining = right - left;
        svbool_t remainingMask = SortSVE::getTrueFalseMask32(remaining);
        svint32_t val = svld1_s32(remainingMask,&array[left]);
        svint32_t val_val = svld1_s32(remainingMask,&values[left]);
        left = right;

        svbool_t mask_low = svcmple_s32(remainingMask, val, pivotvec);
        svbool_t mask_high = svnot_b_z(remainingMask, mask_low);

        const IndexType nb_low = svcntp_b32(remainingMask,mask_low);
        const IndexType nb_high = svcntp_b32(remainingMask,mask_high);

        svint32_t val_low = svcompact_s32(mask_low, val);
        svint32_t val_val_low = svcompact_s32(mask_low, val_val);
        svint32_t val_high = svcompact_s32(mask_high, val);
        svint32_t val_val_high = svcompact_s32(mask_high, val_val);

        mask_low = SortSVE::getTrueFalseMask32(nb_low);
        svst1_s32(mask_low,&array[left_w],val_low);
        svst1_s32(mask_low,&values[left_w],val_val_low);
        left_w += nb_low;

        right_w -= nb_high;
        mask_high = SortSVE::getTrueFalseMask32(nb_high);
        svst1_s32(mask_high,&array[right_w],val_high);
        svst1_s32(mask_high,&values[right_w],val_val_high);
    }
    {
        svbool_t mask = svcmple_s32(svptrue_b32(), left_val, pivotvec);

        const IndexType nb_low = svcntp_b32(svptrue_b32(),mask);
        const IndexType nb_high = S-nb_low;

        svint32_t val_low = svcompact_s32(mask, left_val);
        svint32_t val_val_low = svcompact_s32(mask, left_val_val);
        svbool_t mask_comp_high = svnot_b_z(svptrue_b32(), mask);
        svint32_t val_high = svcompact_s32(mask_comp_high, left_val);
        svint32_t val_val_high = svcompact_s32(mask_comp_high, left_val_val);

        svbool_t mask_low = SortSVE::getTrueFalseMask32(nb_low);
        svst1_s32(mask_low,&array[left_w],val_low);
        svst1_s32(mask_low,&values[left_w],val_val_low);
        left_w += nb_low;

        right_w -= nb_high;
        svbool_t mask_high = SortSVE::getTrueFalseMask32(nb_high);
        svst1_s32(mask_high,&array[right_w],val_high);
        svst1_s32(mask_high,&values[right_w],val_val_high);
    }
    {
        svbool_t mask = svcmple_s32(svptrue_b32(), right_val, pivotvec);

        const IndexType nb_low = svcntp_b32(svptrue_b32(),mask);
        const IndexType nb_high = S-nb_low;

        svint32_t val_low = svcompact_s32(mask, right_val);
        svint32_t val_val_low = svcompact_s32(mask, right_val_val);
        svbool_t mask_comp_high = svnot_b_z(svptrue_b32(), mask);
        svint32_t val_high = svcompact_s32(mask_comp_high, right_val);
        svint32_t val_val_high = svcompact_s32(mask_comp_high, right_val_val);

        svbool_t mask_low = SortSVE::getTrueFalseMask32(nb_low);
        svst1_s32(mask_low,&array[left_w],val_low);
        svst1_s32(mask_low,&values[left_w],val_val_low);
        left_w += nb_low;

        right_w -= nb_high;
        svbool_t mask_high = SortSVE::getTrueFalseMask32(nb_high);
        svst1_s32(mask_high,&array[right_w],val_high);
        svst1_s32(mask_high,&values[right_w],val_val_high);
    }
    return left_w;
}


/* a sequential qs */
template <class IndexType>
static inline IndexType PartitionSVE(std::pair<int,int> array[], IndexType left, IndexType right,
                                     const int pivot){
    const IndexType S = svcntw();

    if(right-left+1 < 2*S){
        return CoreScalarPartition<int,IndexType>(array, left, right, pivot);
    }

    svint32_t pivotvec = svdup_s32(pivot);

    svint32x2_t left_val = svld2_s32(svptrue_b32(),(int*)&array[left]);
    IndexType left_w = left;
    left += S;

    IndexType right_w = right+1;
    right -= S-1;
    svint32x2_t right_val = svld2_s32(svptrue_b32(),(int*)&array[right]);

    while(left + S <= right){
        const IndexType free_left = left - left_w;
        const IndexType free_right = right_w - right;

        svint32x2_t val;
        if( free_left <= free_right ){
            val = svld2_s32(svptrue_b32(),(int*)&array[left]);
            left += S;
        }
        else{
            right -= S;
            val = svld2_s32(svptrue_b32(),(int*)&array[right]);
        }

        svbool_t mask = svcmple_s32(svptrue_b32(), svget2_s32(val, 0), pivotvec);

        const IndexType nb_low = svcntp_b32(svptrue_b32(),mask);
        const IndexType nb_high = S-nb_low;

        svint32x2_t val_low;
        val_low = svset2_s32(val_low, 0,  svcompact_s32(mask, svget2_s32(val, 0)));
        val_low = svset2_s32(val_low, 1,  svcompact_s32(mask, svget2_s32(val, 1)));
        svbool_t mask_comp_high = svnot_b_z(svptrue_b32(), mask);
        svint32x2_t val_high;
        val_high = svset2_s32(val_high, 0,  svcompact_s32(mask_comp_high, svget2_s32(val, 0)));
        val_high = svset2_s32(val_high, 1,  svcompact_s32(mask_comp_high, svget2_s32(val, 1)));

        svbool_t mask_low = SortSVE::getTrueFalseMask32(nb_low);
        svst2_s32(mask_low,(int*)&array[left_w],val_low);
        left_w += nb_low;

        right_w -= nb_high;
        svbool_t mask_high = SortSVE::getTrueFalseMask32(nb_high);
        svst2_s32(mask_high,(int*)&array[right_w],val_high);
    }

    {
        const IndexType remaining = right - left;
        svbool_t remainingMask = SortSVE::getTrueFalseMask32(remaining);
        svint32x2_t val = svld2_s32(remainingMask,(int*)&array[left]);
        left = right;

        svbool_t mask_low = svcmple_s32(remainingMask, svget2_s32(val, 0), pivotvec);
        svbool_t mask_high = svnot_b_z(remainingMask, mask_low);

        const IndexType nb_low = svcntp_b32(remainingMask,mask_low);
        const IndexType nb_high = svcntp_b32(remainingMask,mask_high);

        svint32x2_t val_low;
        val_low = svset2_s32(val_low, 0,  svcompact_s32(mask_low, svget2_s32(val, 0)));
        val_low = svset2_s32(val_low, 1,  svcompact_s32(mask_low, svget2_s32(val, 1)));
        svint32x2_t val_high;
        val_high = svset2_s32(val_high, 0,  svcompact_s32(mask_high, svget2_s32(val, 0)));
        val_high = svset2_s32(val_high, 1,  svcompact_s32(mask_high, svget2_s32(val, 1)));

        mask_low = SortSVE::getTrueFalseMask32(nb_low);
        svst2_s32(mask_low,(int*)&array[left_w],val_low);
        left_w += nb_low;

        right_w -= nb_high;
        mask_high = SortSVE::getTrueFalseMask32(nb_high);
        svst2_s32(mask_high,(int*)&array[right_w],val_high);
    }
    {
        svbool_t mask = svcmple_s32(svptrue_b32(), svget2_s32(left_val, 0), pivotvec);

        const IndexType nb_low = svcntp_b32(svptrue_b32(),mask);
        const IndexType nb_high = S-nb_low;

        svint32x2_t val_low;
        val_low = svset2_s32(val_low, 0,  svcompact_s32(mask, svget2_s32(left_val, 0)));
        val_low = svset2_s32(val_low, 1,  svcompact_s32(mask, svget2_s32(left_val, 1)));
        svbool_t mask_comp_high = svnot_b_z(svptrue_b32(), mask);
        svint32x2_t val_high;
        val_high = svset2_s32(val_high, 0,  svcompact_s32(mask_comp_high, svget2_s32(left_val, 0)));
        val_high = svset2_s32(val_high, 1,  svcompact_s32(mask_comp_high, svget2_s32(left_val, 1)));

        svbool_t mask_low = SortSVE::getTrueFalseMask32(nb_low);
        svst2_s32(mask_low,(int*)&array[left_w],val_low);
        left_w += nb_low;

        right_w -= nb_high;
        svbool_t mask_high = SortSVE::getTrueFalseMask32(nb_high);
        svst2_s32(mask_high,(int*)&array[right_w],val_high);
    }
    {
        svbool_t mask = svcmple_s32(svptrue_b32(), svget2_s32(right_val, 0), pivotvec);

        const IndexType nb_low = svcntp_b32(svptrue_b32(),mask);
        const IndexType nb_high = S-nb_low;

        svint32x2_t val_low;
        val_low = svset2_s32(val_low, 0,  svcompact_s32(mask, svget2_s32(right_val, 0)));
        val_low = svset2_s32(val_low, 1,  svcompact_s32(mask, svget2_s32(right_val, 1)));
        svbool_t mask_comp_high = svnot_b_z(svptrue_b32(), mask);
        svint32x2_t val_high;
        val_high = svset2_s32(val_high, 0,  svcompact_s32(mask_comp_high, svget2_s32(right_val, 0)));
        val_high = svset2_s32(val_high, 1,  svcompact_s32(mask_comp_high, svget2_s32(right_val, 1)));

        svbool_t mask_low = SortSVE::getTrueFalseMask32(nb_low);
        svst2_s32(mask_low,(int*)&array[left_w],val_low);
        left_w += nb_low;

        right_w -= nb_high;
        svbool_t mask_high = SortSVE::getTrueFalseMask32(nb_high);
        svst2_s32(mask_high,(int*)&array[right_w],val_high);
    }
    return left_w;
}

////////////////////////////////////////////////////////////////////////////////
/// Main functions
////////////////////////////////////////////////////////////////////////////////

template <class SortType, class IndexType = size_t>
static inline IndexType CoreSortGetPivot(const SortType array[], const IndexType left, const IndexType right){
    const IndexType nbSteps = 5;
    const IndexType step = std::max(IndexType(1),(right+1-left+nbSteps-1)/nbSteps);
    SortType values[nbSteps] = {array[left], array[step+left], array[2*step+left], array[3*step+left], array[right]};
    IndexType indexes[nbSteps] = {left, step+left, 2*step+left, 3*step+left,right};
    IndexType medianStep[nbSteps];
    for(int idxStep = 0 ; idxStep < 3 ; ++idxStep){
        if(values[idxStep] <= values[idxStep+1] && values[idxStep+1] <= values[idxStep+2]){
            medianStep[idxStep] = idxStep+1;
        }
        else if(values[idxStep+1] <= values[idxStep] && values[idxStep] <= values[idxStep+2]){
            medianStep[idxStep] = idxStep;
        }
        else medianStep[idxStep] = idxStep+2;   
    }
    if(values[medianStep[0]] <= values[medianStep[1]] && values[medianStep[1]] <= values[medianStep[2]]){
        return indexes[medianStep[1]];
    }
    else if(values[medianStep[1]] <= values[medianStep[0]] && values[medianStep[0]] <= values[medianStep[2]]){
        return indexes[medianStep[0]];
    }
    else return indexes[medianStep[2]];
}

template <class SortType, class IndexType = size_t>
static inline IndexType CoreSortGetPivot(const std::pair<SortType,SortType> array[], const IndexType left, const IndexType right){
    const IndexType nbSteps = 5;
    const IndexType step = std::max(IndexType(1),(right+1-left+nbSteps-1)/nbSteps);
    SortType values[nbSteps] = {array[left].first, array[step+left].first, array[2*step+left].first, array[3*step+left].first, array[right].first};
    IndexType indexes[nbSteps] = {left, step+left, 2*step+left, 3*step+left,right};
    IndexType medianStep[nbSteps];
    for(int idxStep = 0 ; idxStep < 3 ; ++idxStep){
        if(values[idxStep] <= values[idxStep+1] && values[idxStep+1] <= values[idxStep+2]){
            medianStep[idxStep] = idxStep+1;
        }
        else if(values[idxStep+1] <= values[idxStep] && values[idxStep] <= values[idxStep+2]){
            medianStep[idxStep] = idxStep;
        }
        else medianStep[idxStep] = idxStep+2;   
    }
    if(values[medianStep[0]] <= values[medianStep[1]] && values[medianStep[1]] <= values[medianStep[2]]){
        return indexes[medianStep[1]];
    }
    else if(values[medianStep[1]] <= values[medianStep[0]] && values[medianStep[0]] <= values[medianStep[2]]){
        return indexes[medianStep[0]];
    }
    else return indexes[medianStep[2]];
}

template <class SortType, class IndexType = size_t>
static inline IndexType CoreSortPivotPartition(SortType array[], const IndexType left, const IndexType right){
    if(right-left > 1){
        const IndexType pivotIdx = CoreSortGetPivot(array, left, right);
        std::swap(array[pivotIdx], array[right]);
        const IndexType part = PartitionSVE(array, left, right-1, array[right].first);
        std::swap(array[part], array[right]);
        return part;
    }
    return left;
}

template <class SortType, class IndexType = size_t>
static inline IndexType CoreSortPivotPartition(SortType array[], SortType values[], const IndexType left, const IndexType right){
    if(right-left > 1){
        const IndexType pivotIdx = CoreSortGetPivot(array, left, right);
        std::swap(array[pivotIdx], array[right]);
        std::swap(values[pivotIdx], values[right]);
        const IndexType part = PartitionSVE(array, values, left, right-1, array[right]);
        std::swap(array[part], array[right]);
        std::swap(values[part], values[right]);
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
static inline IndexType CoreSortPartition(SortType array[], SortType values[],  const IndexType left, const IndexType right,
                                          const SortType pivot){
    return  PartitionSVE(array, values, left, right, pivot);
}

template <class SortType, class IndexType = size_t>
static void CoreSort(SortType array[], const IndexType left, const IndexType right){
    static const IndexType SortLimite = 16*svcntb()/(sizeof(SortType)/2);
    if(right-left < SortLimite){
        SmallSort16V(array+left, right-left+1);
    }
    else{
        const IndexType part = CoreSortPivotPartition<SortType,IndexType>(array, left, right);
        if(part+1 < right) CoreSort<SortType,IndexType>(array,part+1,right);
        if(part && left < part-1)  CoreSort<SortType,IndexType>(array,left,part - 1);
    }
}

template <class SortType, class IndexType = size_t>
static void CoreSort(SortType array[], SortType values[], const IndexType left, const IndexType right){
    static const IndexType SortLimite = 16*svcntb()/sizeof(SortType);
    if(right-left < SortLimite){
        SmallSort16V(array+left, values+left, right-left+1);
    }
    else{
        const IndexType part = CoreSortPivotPartition<SortType,IndexType>(array, values, left, right);
        if(part+1 < right) CoreSort<SortType,IndexType>(array,values,part+1,right);
        if(part && left < part-1)  CoreSort<SortType,IndexType>(array,values,left,part - 1);
    }
}

template <class SortType, class IndexType = size_t>
static inline void Sort(SortType array[], const IndexType size){
    static const IndexType SortLimite = 16*svcntb()/(sizeof(SortType)/2);
    if(size <= SortLimite){
        SmallSort16V(array, size);
        return;
    }
    CoreSort<SortType,IndexType>(array, 0, size-1);
}

template <class SortType, class IndexType = size_t>
static inline void Sort(SortType array[], SortType values[], const IndexType size){
    const IndexType SortLimite = 16*svcntb()/sizeof(SortType);
    if(size <= SortLimite){
        SmallSort16V(array, values, size);
        return;
    }
    CoreSort<SortType,IndexType>(array, values, 0, size-1);
}


#if defined(_OPENMP)


template <class SortType, class IndexType = size_t>
static inline void CoreSortTask(SortType array[], const IndexType left, const IndexType right, const int deep){
    static const IndexType SortLimite = 16*svcntb()/(sizeof(SortType)/2);
    if(right-left < SortLimite){
        SmallSort16V(array+left, right-left+1);
    }
    else{
        const IndexType part = CoreSortPivotPartition<SortType,IndexType>(array, left, right);
        if( deep > 1 && part+1 < right && part && left < part-1){
            if(right-part > part-left){
                // default(none) has been removed for clang compatibility
                #pragma omp task default(shared) firstprivate(array, part, left, deep) priority(deep)
                CoreSortTask<SortType,IndexType>(array,left,part - 1, deep-1);
                // not task needed, let the current thread compute it
                CoreSortTask<SortType,IndexType>(array,part+1,right, deep-1);
            }                                        
            else{
                // default(none) has been removed for clang compatibility
                #pragma omp task default(shared) firstprivate(array, part, right, deep) 
                CoreSortTask<SortType,IndexType>(array,part+1,right, deep-1);
                // not task needed, let the current thread compute it
                CoreSortTask<SortType,IndexType>(array,left,part - 1, deep-1);
            }
        }
        else {
            if(part+1 < right) CoreSort<SortType,IndexType>(array, part+1,right);
            if(part && left < part-1)  CoreSort<SortType,IndexType>(array, left,part - 1);
        }
    }
}

template <class SortType, class IndexType = size_t>
static inline void CoreSortTask(SortType array[], SortType values[], const IndexType left, const IndexType right, const int deep){
    static const IndexType SortLimite = 16*svcntb()/sizeof(SortType);
    if(right-left < SortLimite){
        SmallSort16V(array+left, values+left, right-left+1);
    }
    else{
        const IndexType part = CoreSortPivotPartition<SortType,IndexType>(array, values, left, right);
        if( deep > 1 && part+1 < right && part && left < part-1){
            if(right-part > part-left){
                // default(none) has been removed for clang compatibility
                #pragma omp task default(shared) firstprivate(array, values, part, left, deep) priority(deep)
                CoreSortTask<SortType,IndexType>(array,values,left,part - 1, deep-1);
                // not task needed, let the current thread compute it
                CoreSortTask<SortType,IndexType>(array,values,part+1,right, deep-1);
            }                                        
            else{
                // default(none) has been removed for clang compatibility
                #pragma omp task default(shared) firstprivate(array, values, part, right, deep) 
                CoreSortTask<SortType,IndexType>(array,values,part+1,right, deep-1);
                // not task needed, let the current thread compute it
                CoreSortTask<SortType,IndexType>(array,values,left,part - 1, deep-1);
            }
        }
        else {
            if(part+1 < right) CoreSort<SortType,IndexType>(array,values, part+1,right);
            if(part && left < part-1)  CoreSort<SortType,IndexType>(array,values, left,part - 1);
        }
    }
}

template <class SortType, class IndexType = size_t>
static inline void SortOmpPartition(SortType array[], const IndexType size){
    const int nbTasksRequiere = (omp_get_max_threads() * 5);
    int deep = 0;
    while( (1 << deep) < nbTasksRequiere ) deep += 1;

#pragma omp parallel
    {
#pragma omp master
        {
            CoreSortTask<SortType,IndexType>(array, 0, size - 1 , deep);
        }
    }
}

template <class SortType, class IndexType = size_t>
static inline void SortOmpPartition(SortType array[], SortType values[], const IndexType size){
    const int nbTasksRequiere = (omp_get_max_threads() * 5);
    int deep = 0;
    while( (1 << deep) < nbTasksRequiere ) deep += 1;

#pragma omp parallel
    {
#pragma omp master
        {
            CoreSortTask<SortType,IndexType>(array, values, 0, size - 1 , deep);
        }
    }
}
#endif

}


#endif
