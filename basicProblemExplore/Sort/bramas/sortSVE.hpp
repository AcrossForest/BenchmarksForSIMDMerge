//////////////////////////////////////////////////////////
/// By berenger.bramas@inria.fr 2020.
/// Licence is MIT.
/// Comes without any warranty.
///
/// Code to sort an array of integer or double
/// using ARM SVE (works for vectors of any size).
/// It also includes a partitioning function.
///
/// Please refer to the README to know how to build
/// and to have more information about the functions.
///
//////////////////////////////////////////////////////////
#ifndef SORTSVE_HPP
#define SORTSVE_HPP

#ifndef __ARM_FEATURE_SVE
#warning __ARM_FEATURE_SVE undefined
#endif
#include <arm_sve.h>

#include <climits>
#include <cfloat>
#include <algorithm>
#include <cstdio>
#include <deque>


#if defined(_OPENMP)
#include <omp.h>
#include <cassert>
#else
#warning OpenMP disabled
#endif

namespace SortSVE{

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

inline void CoreExchange(svint32_t& input, svint32_t& input2){
    svint32_t permNeigh2 = svrev_s32(input2);
    input2 = svmax_s32_z(svptrue_b32(), input, permNeigh2);
    input = svmin_s32_z(svptrue_b32(), input, permNeigh2);
}

inline void CoreExchangeEnd(svint32_t& input, svint32_t& input2){
    svint32_t inputCopy = input;
    input = svmin_s32_z(svptrue_b32(), input, input2);
    input2 = svmax_s32_z(svptrue_b32(), inputCopy, input2);
}


inline void CoreExchange(svfloat64_t& input, svfloat64_t& input2){
    svfloat64_t permNeigh2 = svrev_f64(input2);
    input2 = svmax_f64_z(svptrue_b64(), input, permNeigh2);
    input = svmin_f64_z(svptrue_b64(), input, permNeigh2);
}

inline void CoreExchangeEnd(svfloat64_t& input, svfloat64_t& input2){
    svfloat64_t inputCopy = input;
    input = svmin_f64_z(svptrue_b64(), input, input2);
    input2 = svmax_f64_z(svptrue_b64(), inputCopy, input2);
}

inline bool IsSorted(const svint32_t& input){
//    // Methode 1: 1 vec op, 1 comp, 2 bool vec op, 2 bool vec count
//    // Inverse : 0 1 2 3 gives 3 2 1 0
//    svint32_t revinput = svrev_s32(input);
//    // Compare: 0 1 2 3 > 3 2 1 0 gives F F T T
//    svbool_t mask = svcmpgt_s32(svptrue_b32(), input, revinput);
//    // Brka : F F T T give T T F F
//    svbool_t v1100 = svbrkb_b_z(svptrue_b32(), mask);
//    // Inv(Brka) should be the same
//    return svcntp_b32(svptrue_b32(),svnot_b_z(svptrue_b32(),v1100)) == svcntp_b32(svptrue_b32(),mask);
    // Methode 2: 1 vec op, 1 comp, 2 bool vec op, 1 bool vec count
    svbool_t FTTT = getFalseTrueMask32(1);
    svint32_t compactinput = svcompact_s32(FTTT, input);
    const size_t vecSizeM1 = (svcntb()/sizeof(int))-1;
    svbool_t TTTF = getTrueFalseMask32(vecSizeM1);
    svbool_t mask = svcmple_s32(svptrue_b32(), input, compactinput);
    return svcntp_b32(TTTF,mask) == vecSizeM1;
}

inline bool IsSorted(const svfloat64_t& input){
//    // Methode 1: 1 vec op, 1 comp, 2 bool vec op, 2 bool vec count
//    // Inverse : 0 1 2 3 gives 3 2 1 0
//    svfloat64_t revinput = svrev_f64(input);
//    // Compare: 0 1 2 3 > 3 2 1 0 gives F F T T
//    svbool_t mask = svcmpgt_f64(svptrue_b64(), input, revinput);
//    // Brka : F F T T give T T F F
//    svbool_t v1100 = svbrkb_b_z(svptrue_b64(), mask);
//    // Inv(Brka) should be the same
//    return svcntp_b32(svptrue_b64(),svnot_b_z(svptrue_b64(),v1100)) == svcntp_b64(svptrue_b64(),mask);
    svbool_t FTTT = getFalseTrueMask64(1);
    svfloat64_t compactinput = svcompact_f64(FTTT, input);
    const size_t vecSizeM1 = (svcntb()/sizeof(double))-1;
    svbool_t TTTF = getTrueFalseMask64(vecSizeM1);
    svbool_t mask = svcmple_f64(svptrue_b64(), input, compactinput);
    return svcntp_b64(TTTF,mask) == vecSizeM1;
}


inline void CoreSmallSort(svint32_t& input){
    if(IsSorted(input)){
        return;
    }

    const int nbValuesInVec = svcntw();

    const svint32_t vecindex = svindex_s32(0, 1);

    svbool_t ffttout = svzip1_b32(svpfalse_b(),svptrue_b32());

    svint32_t vecincout = svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

    {// stepout == 1
        const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

        const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));

        input = svsel_s32(ffttout,
                          svmax_s32_z(svptrue_b32(), input, inputPerm),
                          svmin_s32_z(svptrue_b32(), input, inputPerm));
    }
    for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
        ffttout = svzip1_b32(ffttout,ffttout);
        vecincout = svsel_s32(ffttout,
                                  svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                                  svadd_n_s32_z(svptrue_b32(), vecincout, stepout));
        {
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
        }

        svbool_t fftt = svuzp2_b32(ffttout,ffttout);

        svint32_t vecinc = svdup_s32(stepout/2);

        for(long int step = stepout/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast

            if(IsSorted(input)){
                return;
            }
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
        }
    }
}


inline void CoreSmallSortEnd(svint32_t& input){
    if(IsSorted(input)){
        return;
    }

    const int nbValuesInVec = svcntw();

    svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

    const svint32_t vecindex = svindex_s32(0, 1);

    svint32_t vecinc = svdup_s32(nbValuesInVec/2);

    bool isSorted = IsSorted(input);
    for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
        const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

        const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));

        input = svsel_s32(fftt,
                          svmax_s32_z(svptrue_b32(), input, inputPerm),
                          svmin_s32_z(svptrue_b32(), input, inputPerm));

        fftt = svuzp2_b32(fftt,fftt);
        vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast

        if(IsSorted(input)){
            return;
        }
    }
    { // Step == 1
        const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

        const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));

        input = svsel_s32(fftt,
                          svmax_s32_z(svptrue_b32(), input, inputPerm),
                          svmin_s32_z(svptrue_b32(), input, inputPerm));
    }
}


inline void CoreSmallSort(svfloat64_t& input){    
    if(IsSorted(input)){
        return;
    }
    const int nbValuesInVec = svcntd();

    const svint64_t vecindex = svindex_s64(0, 1);

    svbool_t ffttout = svzip1_b64(svpfalse_b(),svptrue_b64());

    svint64_t vecincout = svsel_s64(ffttout, svdup_s64(-1), svdup_s64(1));

    {// stepout == 1
        const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, vecincout);

        const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));

        input = svsel_f64(ffttout,
                          svmax_f64_z(svptrue_b64(), input, inputPerm),
                          svmin_f64_z(svptrue_b64(), input, inputPerm));
    }
    for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
        ffttout = svzip1_b64(ffttout,ffttout);
        vecincout = svsel_s64(ffttout,
                                  svsub_n_s64_z(svptrue_b64(), vecincout, stepout),
                                  svadd_n_s64_z(svptrue_b64(), vecincout, stepout));
        {
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, vecincout);

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));
        }

        svbool_t fftt = svuzp2_b64(ffttout,ffttout);

        svint64_t vecinc = svdup_s64(stepout/2);

        for(long int step = stepout/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast

            if(IsSorted(input)){
                return;
            }
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));
        }
    }
}

inline void CoreSmallSortEnd(svfloat64_t& input){    
    if(IsSorted(input)){
        return;
    }
    const int nbValuesInVec = svcntd();

    svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

    const svint64_t vecindex = svindex_s64(0, 1);

    svint64_t vecinc = svdup_s64(nbValuesInVec/2);

    for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
        const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

        const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));

        input = svsel_f64(fftt,
                          svmax_f64_z(svptrue_b64(), input, inputPerm),
                          svmin_f64_z(svptrue_b64(), input, inputPerm));

        fftt = svuzp2_b64(fftt,fftt);
        vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast

        if(IsSorted(input)){
            return;
        }
    }
    { // Step == 1
        const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

        const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));

        input = svsel_f64(fftt,
                          svmax_f64_z(svptrue_b64(), input, inputPerm),
                          svmin_f64_z(svptrue_b64(), input, inputPerm));
    }
}


inline void CoreSmallEnd2(svfloat64_t& input, svfloat64_t& input2){
    if(svmaxv_f64(svptrue_b64(),input) > svminv_f64(svptrue_b64(),input2)){
        CoreExchangeEnd(input, input2);
    }
#ifdef NOOPTIM
    CoreSmallSortEnd(input);
    CoreSmallSortEnd(input2);
#else
    bool isSorted1 = IsSorted(input);
    bool isSorted2 = IsSorted(input2);
    if(!isSorted1 && !isSorted2){
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
            isSorted1 |= IsSorted(input);
            isSorted2 |= IsSorted(input2);
            if(isSorted1 && isSorted2){
                return;
            }
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));
        }
    }
    else if(!isSorted1){
        CoreSmallSortEnd(input);
    }
    else if(!isSorted2){
        CoreSmallSortEnd(input2);
    }
#endif
}

inline void CoreSmallSort2(svfloat64_t& input, svfloat64_t& input2){
#ifdef NOOPTIM
    CoreSmallSort(input);
    CoreSmallSort(input2);
#else
    bool isSorted1 = IsSorted(input);
    bool isSorted2 = IsSorted(input2);
    if(!isSorted1 && !isSorted2){
        const int nbValuesInVec = svcntd();

        const svint64_t vecindex = svindex_s64(0, 1);

        svbool_t ffttout = svzip1_b64(svpfalse_b(),svptrue_b64());

        svint64_t vecincout = svsel_s64(ffttout, svdup_s64(-1), svdup_s64(1));

        {// stepout == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, vecincout);

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));


            input2 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input2, input2Perm),
                              svmin_f64_z(svptrue_b64(), input2, input2Perm));
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec && !(isSorted1 && isSorted2) ; stepout *= 2){
            ffttout = svzip1_b64(ffttout,ffttout);
            vecincout = svsel_s64(ffttout,
                                      svsub_n_s64_z(svptrue_b64(), vecincout, stepout),
                                      svadd_n_s64_z(svptrue_b64(), vecincout, stepout));
            {
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, vecincout);

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));


                input2 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));
            }

            svbool_t fftt = svuzp2_b64(ffttout,ffttout);

            svint64_t vecinc = svdup_s64(stepout/2);

            for(long int step = stepout/2 ; step > 1 && !(isSorted1 && isSorted2) ; step/=2){
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));


                input2 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));

                fftt = svuzp2_b64(fftt,fftt);
                vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast

                isSorted1 |= IsSorted(input);
                isSorted2 |= IsSorted(input2);
            }
            { // Step == 1
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));


                input2 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));
            }
        }
    }
    else if(!isSorted1){
        CoreSmallSort(input);
    }
    else if(!isSorted2){
        CoreSmallSort(input2);
    }
#endif
    if(svmaxv_f64(svptrue_b64(),input) > svminv_f64(svptrue_b64(),input2)){
        CoreExchange(input, input2);
    }
#ifdef NOOPTIM
    CoreSmallSortEnd(input);
    CoreSmallSortEnd(input2);
#else
    isSorted1 = IsSorted(input);
    isSorted2 = IsSorted(input2);
    if(!isSorted1 && !isSorted2){
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
            isSorted1 |= IsSorted(input);
            isSorted2 |= IsSorted(input2);
            if(isSorted1 && isSorted2){
                return;
            }
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));
        }
    }
    else if(!isSorted1){
        CoreSmallSortEnd(input);
    }
    else if(!isSorted2){
        CoreSmallSortEnd(input2);
    }
#endif
}

inline void CoreSmallEnd3(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3){
    {
        CoreExchangeEnd(input, input3);
    }
#ifdef NOOPTIM
    CoreSmallEnd2(input, input2);
    CoreSmallSortEnd(input3);
#else
    {
        CoreExchangeEnd(input, input2);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));
        }
    }
#endif
}

inline void CoreSmallSort3(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3){
#ifdef NOOPTIM
    CoreSmallSort2(input, input2);
    CoreSmallSort(input3);
#else
    {
        const int nbValuesInVec = svcntd();

        const svint64_t vecindex = svindex_s64(0, 1);

        svbool_t ffttout = svzip1_b64(svpfalse_b(),svptrue_b64());

        svint64_t vecincout = svsel_s64(ffttout, svdup_s64(-1), svdup_s64(1));

        {// stepout == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, vecincout);

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));


            input2 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input2, input2Perm),
                              svmin_f64_z(svptrue_b64(), input2, input2Perm));


            input3 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input3, input3Perm),
                              svmin_f64_z(svptrue_b64(), input3, input3Perm));
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b64(ffttout,ffttout);
            vecincout = svsel_s64(ffttout,
                                      svsub_n_s64_z(svptrue_b64(), vecincout, stepout),
                                      svadd_n_s64_z(svptrue_b64(), vecincout, stepout));
            {
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, vecincout);

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));


                input2 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));


                input3 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));
            }

            svbool_t fftt = svuzp2_b64(ffttout,ffttout);

            svint64_t vecinc = svdup_s64(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));

                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));

                input2 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));


                input3 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));

                fftt = svuzp2_b64(fftt,fftt);
                vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));

                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));

                input2 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));


                input3 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));
            }
        }
    }
    {
        CoreExchange(input, input2);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));
        }
    }
#endif
    {
        CoreExchange(input2, input3);
    }
#ifdef NOOPTIM
    CoreSmallEnd2(input, input2);
    CoreSmallSortEnd(input3);
#else
    {
        CoreExchangeEnd(input, input2);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));
        }
    }
#endif
}

inline void CoreSmallEnd4(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4){
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
#ifdef NOOPTIM
    CoreSmallEnd2(input, input2);
    CoreSmallEnd2(input3, input4);
#else
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));
        }
    }
#endif
}

inline void CoreSmallSort4(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4){
#ifdef NOOPTIM
    CoreSmallSort2(input, input2);
    CoreSmallSort2(input3, input4);
#else
    {
        const int nbValuesInVec = svcntd();

        const svint64_t vecindex = svindex_s64(0, 1);

        svbool_t ffttout = svzip1_b64(svpfalse_b(),svptrue_b64());

        svint64_t vecincout = svsel_s64(ffttout, svdup_s64(-1), svdup_s64(1));

        {// stepout == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, vecincout);

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));


            input2 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input2, input2Perm),
                              svmin_f64_z(svptrue_b64(), input2, input2Perm));

            const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input3 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input3, input3Perm),
                              svmin_f64_z(svptrue_b64(), input3, input3Perm));


            input4 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input4, input4Perm),
                              svmin_f64_z(svptrue_b64(), input4, input4Perm));
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b64(ffttout,ffttout);
            vecincout = svsel_s64(ffttout,
                                      svsub_n_s64_z(svptrue_b64(), vecincout, stepout),
                                      svadd_n_s64_z(svptrue_b64(), vecincout, stepout));
            {
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, vecincout);

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));


                input2 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));

                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

                input3 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));


                input4 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input4, input4Perm),
                                  svmin_f64_z(svptrue_b64(), input4, input4Perm));
            }

            svbool_t fftt = svuzp2_b64(ffttout,ffttout);

            svint64_t vecinc = svdup_s64(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));


                input2 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));

                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

                input3 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));


                input4 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input4, input4Perm),
                                  svmin_f64_z(svptrue_b64(), input4, input4Perm));

                fftt = svuzp2_b64(fftt,fftt);
                vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));


                input2 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));

                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

                input3 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));


                input4 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input4, input4Perm),
                                  svmin_f64_z(svptrue_b64(), input4, input4Perm));
            }
        }
    }
    {
        CoreExchange(input, input2);
        CoreExchange(input3, input4);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, input3Perm),
                              svmin_f64_z(svptrue_b64(), input3, input3Perm));


            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, input4Perm),
                              svmin_f64_z(svptrue_b64(), input4, input4Perm));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, input3Perm),
                              svmin_f64_z(svptrue_b64(), input3, input3Perm));


            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, input4Perm),
                              svmin_f64_z(svptrue_b64(), input4, input4Perm));
        }
    }
#endif
    {
        CoreExchange(input, input4);
        CoreExchange(input2, input3);
    }
#ifdef NOOPTIM
    CoreSmallEnd2(input, input2);
    CoreSmallEnd2(input3, input4);
#else
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));
        }
    }
#endif
}

inline void CoreSmallEnd5(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4, svfloat64_t& input5){
    {
        CoreExchangeEnd(input, input5);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4);
    CoreSmallSortEnd(input5);
#else
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm5 = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, inputPerm5),
                              svmin_f64_z(svptrue_b64(), input5, inputPerm5));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm5 = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, inputPerm5),
                              svmin_f64_z(svptrue_b64(), input5, inputPerm5));
        }
    }
#endif
}

inline void CoreSmallSort5(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                           svfloat64_t& input5){
#ifdef NOOPTIM
    CoreSmallSort4(input, input2, input3, input4);
    CoreSmallSort(input5);
#else
    {
        const int nbValuesInVec = svcntd();

        const svint64_t vecindex = svindex_s64(0, 1);

        svbool_t ffttout = svzip1_b64(svpfalse_b(),svptrue_b64());

        svint64_t vecincout = svsel_s64(ffttout, svdup_s64(-1), svdup_s64(1));

        {// stepout == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, vecincout);

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));


            input2 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input2, input2Perm),
                              svmin_f64_z(svptrue_b64(), input2, input2Perm));


            input3 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input3, input3Perm),
                              svmin_f64_z(svptrue_b64(), input3, input3Perm));

            const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));

            input4 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input4, input4Perm),
                              svmin_f64_z(svptrue_b64(), input4, input4Perm));


            input5 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input5, input5Perm),
                              svmin_f64_z(svptrue_b64(), input5, input5Perm));
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b64(ffttout,ffttout);
            vecincout = svsel_s64(ffttout,
                                      svsub_n_s64_z(svptrue_b64(), vecincout, stepout),
                                      svadd_n_s64_z(svptrue_b64(), vecincout, stepout));
            {
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, vecincout);

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));


                input2 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));


                input3 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));

                const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));

                input4 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input4, input4Perm),
                                  svmin_f64_z(svptrue_b64(), input4, input4Perm));


                input5 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input5, input5Perm),
                                  svmin_f64_z(svptrue_b64(), input5, input5Perm));
            }

            svbool_t fftt = svuzp2_b64(ffttout,ffttout);

            svint64_t vecinc = svdup_s64(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));


                input2 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));

                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));

                input3 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));


                input4 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input4, input4Perm),
                                  svmin_f64_z(svptrue_b64(), input4, input4Perm));


                input5 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input5, input5Perm),
                                  svmin_f64_z(svptrue_b64(), input5, input5Perm));

                fftt = svuzp2_b64(fftt,fftt);
                vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));


                input2 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));


                input3 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));

                const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));

                input4 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input4, input4Perm),
                                  svmin_f64_z(svptrue_b64(), input4, input4Perm));


                input5 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input5, input5Perm),
                                  svmin_f64_z(svptrue_b64(), input5, input5Perm));
            }
        }
    }
    {
        CoreExchange(input, input2);
        CoreExchange(input3, input4);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, input3Perm),
                              svmin_f64_z(svptrue_b64(), input3, input3Perm));


            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, input4Perm),
                              svmin_f64_z(svptrue_b64(), input4, input4Perm));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, input3Perm),
                              svmin_f64_z(svptrue_b64(), input3, input3Perm));


            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, input4Perm),
                              svmin_f64_z(svptrue_b64(), input4, input4Perm));
        }
    }
    {
        CoreExchange(input, input4);
        CoreExchange(input2, input3);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));
        }
    }
#endif
    {
        CoreExchange(input4, input5);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4);
    CoreSmallSortEnd(input5);
#else
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm5 = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, inputPerm5),
                              svmin_f64_z(svptrue_b64(), input5, inputPerm5));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm5 = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, inputPerm5),
                              svmin_f64_z(svptrue_b64(), input5, inputPerm5));
        }
    }
#endif
}

inline void CoreSmallEnd6(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                          svfloat64_t& input5, svfloat64_t& input6){
    {
        CoreExchangeEnd(input, input5);
        CoreExchangeEnd(input2, input6);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4);
    CoreSmallEnd2(input5, input6);
#else
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchangeEnd(input5, input6);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm5 = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm6 = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, inputPerm5),
                              svmin_f64_z(svptrue_b64(), input5, inputPerm5));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, inputPerm6),
                              svmin_f64_z(svptrue_b64(), input6, inputPerm6));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm5 = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm6 = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, inputPerm5),
                              svmin_f64_z(svptrue_b64(), input5, inputPerm5));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, inputPerm6),
                              svmin_f64_z(svptrue_b64(), input6, inputPerm6));
        }
    }
#endif
}

inline void CoreSmallSort6(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                           svfloat64_t& input5, svfloat64_t& input6){
#ifdef NOOPTIM
    CoreSmallSort4(input, input2, input3, input4);
    CoreSmallSort2(input5, input6);
#else
    {
        const int nbValuesInVec = svcntd();

        const svint64_t vecindex = svindex_s64(0, 1);

        svbool_t ffttout = svzip1_b64(svpfalse_b(),svptrue_b64());

        svint64_t vecincout = svsel_s64(ffttout, svdup_s64(-1), svdup_s64(1));

        {// stepout == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, vecincout);

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));


            input2 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input2, input2Perm),
                              svmin_f64_z(svptrue_b64(), input2, input2Perm));

            const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input3 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input3, input3Perm),
                              svmin_f64_z(svptrue_b64(), input3, input3Perm));


            input4 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input4, input4Perm),
                              svmin_f64_z(svptrue_b64(), input4, input4Perm));

            const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

            input5 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input5, input5Perm),
                              svmin_f64_z(svptrue_b64(), input5, input5Perm));


            input6 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input6, input6Perm),
                              svmin_f64_z(svptrue_b64(), input6, input6Perm));
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b64(ffttout,ffttout);
            vecincout = svsel_s64(ffttout,
                                      svsub_n_s64_z(svptrue_b64(), vecincout, stepout),
                                      svadd_n_s64_z(svptrue_b64(), vecincout, stepout));
            {
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, vecincout);

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));


                input2 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));

                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

                input3 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));


                input4 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input4, input4Perm),
                                  svmin_f64_z(svptrue_b64(), input4, input4Perm));

                const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

                input5 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input5, input5Perm),
                                  svmin_f64_z(svptrue_b64(), input5, input5Perm));


                input6 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input6, input6Perm),
                                  svmin_f64_z(svptrue_b64(), input6, input6Perm));
            }

            svbool_t fftt = svuzp2_b64(ffttout,ffttout);

            svint64_t vecinc = svdup_s64(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));


                input2 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));

                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

                input3 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));


                input4 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input4, input4Perm),
                                  svmin_f64_z(svptrue_b64(), input4, input4Perm));

                const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

                input5 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input5, input5Perm),
                                  svmin_f64_z(svptrue_b64(), input5, input5Perm));


                input6 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input6, input6Perm),
                                  svmin_f64_z(svptrue_b64(), input6, input6Perm));

                fftt = svuzp2_b64(fftt,fftt);
                vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));


                input2 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));

                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

                input3 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));


                input4 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input4, input4Perm),
                                  svmin_f64_z(svptrue_b64(), input4, input4Perm));

                const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

                input5 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input5, input5Perm),
                                  svmin_f64_z(svptrue_b64(), input5, input5Perm));


                input6 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input6, input6Perm),
                                  svmin_f64_z(svptrue_b64(), input6, input6Perm));
            }
        }
    }
    {
        CoreExchange(input, input2);
        CoreExchange(input3, input4);
    }
    {
        CoreExchange(input5, input6);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, input3Perm),
                              svmin_f64_z(svptrue_b64(), input3, input3Perm));


            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, input4Perm),
                              svmin_f64_z(svptrue_b64(), input4, input4Perm));

            const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, input5Perm),
                              svmin_f64_z(svptrue_b64(), input5, input5Perm));


            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, input6Perm),
                              svmin_f64_z(svptrue_b64(), input6, input6Perm));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, input3Perm),
                              svmin_f64_z(svptrue_b64(), input3, input3Perm));


            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, input4Perm),
                              svmin_f64_z(svptrue_b64(), input4, input4Perm));

            const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, input5Perm),
                              svmin_f64_z(svptrue_b64(), input5, input5Perm));


            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, input6Perm),
                              svmin_f64_z(svptrue_b64(), input6, input6Perm));
        }
    }
    {
        CoreExchange(input, input4);
        CoreExchange(input2, input3);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchangeEnd(input5, input6);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, input5Perm),
                              svmin_f64_z(svptrue_b64(), input5, input5Perm));


            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, input6Perm),
                              svmin_f64_z(svptrue_b64(), input6, input6Perm));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, input5Perm),
                              svmin_f64_z(svptrue_b64(), input5, input5Perm));


            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, input6Perm),
                              svmin_f64_z(svptrue_b64(), input6, input6Perm));
        }
    }
#endif
    {
        CoreExchange(input3, input6);
        CoreExchange(input4, input5);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4);
    CoreSmallEnd2(input5, input6);
#else
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchangeEnd(input5, input6);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm5 = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm6 = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, inputPerm5),
                              svmin_f64_z(svptrue_b64(), input5, inputPerm5));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, inputPerm6),
                              svmin_f64_z(svptrue_b64(), input6, inputPerm6));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm5 = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm6 = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, inputPerm5),
                              svmin_f64_z(svptrue_b64(), input5, inputPerm5));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, inputPerm6),
                              svmin_f64_z(svptrue_b64(), input6, inputPerm6));
        }
    }
#endif
}

inline void CoreSmallEnd7(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                          svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7){
    {
        CoreExchangeEnd(input, input5);
        CoreExchangeEnd(input2, input6);
        CoreExchangeEnd(input3, input7);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4);
    CoreSmallEnd3(input5, input6, input7);
#else
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchangeEnd(input5, input7);
    }
    {
        CoreExchangeEnd(input5, input6);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm5 = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm6 = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm7 = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, inputPerm5),
                              svmin_f64_z(svptrue_b64(), input5, inputPerm5));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, inputPerm6),
                              svmin_f64_z(svptrue_b64(), input6, inputPerm6));

            input7 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input7, inputPerm7),
                              svmin_f64_z(svptrue_b64(), input7, inputPerm7));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm5 = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm6 = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm7 = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, inputPerm5),
                              svmin_f64_z(svptrue_b64(), input5, inputPerm5));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, inputPerm6),
                              svmin_f64_z(svptrue_b64(), input6, inputPerm6));

            input7 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input7, inputPerm7),
                              svmin_f64_z(svptrue_b64(), input7, inputPerm7));
        }
    }
#endif
}

inline void CoreSmallSort7(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                           svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7){
#ifdef NOOPTIM
    CoreSmallSort4(input, input2, input3, input4);
    CoreSmallSort3(input5, input6, input7);
#else
    {
        const int nbValuesInVec = svcntd();

        const svint64_t vecindex = svindex_s64(0, 1);

        svbool_t ffttout = svzip1_b64(svpfalse_b(),svptrue_b64());

        svint64_t vecincout = svsel_s64(ffttout, svdup_s64(-1), svdup_s64(1));

        {// stepout == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, vecincout);

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));


            input2 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input2, input2Perm),
                              svmin_f64_z(svptrue_b64(), input2, input2Perm));

            const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input3 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input3, input3Perm),
                              svmin_f64_z(svptrue_b64(), input3, input3Perm));


            input4 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input4, input4Perm),
                              svmin_f64_z(svptrue_b64(), input4, input4Perm));

            const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

            input5 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input5, input5Perm),
                              svmin_f64_z(svptrue_b64(), input5, input5Perm));


            input6 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input6, input6Perm),
                              svmin_f64_z(svptrue_b64(), input6, input6Perm));

            const svfloat64_t input7Perm = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));

            input7 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input7, input7Perm),
                              svmin_f64_z(svptrue_b64(), input7, input7Perm));
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b64(ffttout,ffttout);
            vecincout = svsel_s64(ffttout,
                                      svsub_n_s64_z(svptrue_b64(), vecincout, stepout),
                                      svadd_n_s64_z(svptrue_b64(), vecincout, stepout));
            {
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, vecincout);

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));

                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));

                input2 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));


                input3 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));

                const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));

                input4 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input4, input4Perm),
                                  svmin_f64_z(svptrue_b64(), input4, input4Perm));


                input5 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input5, input5Perm),
                                  svmin_f64_z(svptrue_b64(), input5, input5Perm));

                const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input7Perm = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));

                input6 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input6, input6Perm),
                                  svmin_f64_z(svptrue_b64(), input6, input6Perm));


                input7 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input7, input7Perm),
                                  svmin_f64_z(svptrue_b64(), input7, input7Perm));
            }

            svbool_t fftt = svuzp2_b64(ffttout,ffttout);

            svint64_t vecinc = svdup_s64(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));

                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));

                input2 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));


                input3 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));

                const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));

                input4 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input4, input4Perm),
                                  svmin_f64_z(svptrue_b64(), input4, input4Perm));


                input5 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input5, input5Perm),
                                  svmin_f64_z(svptrue_b64(), input5, input5Perm));

                const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input7Perm = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));

                input6 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input6, input6Perm),
                                  svmin_f64_z(svptrue_b64(), input6, input6Perm));


                input7 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input7, input7Perm),
                                  svmin_f64_z(svptrue_b64(), input7, input7Perm));

                fftt = svuzp2_b64(fftt,fftt);
                vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));

                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));

                input2 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));


                input3 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));

                const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));

                input4 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input4, input4Perm),
                                  svmin_f64_z(svptrue_b64(), input4, input4Perm));


                input5 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input5, input5Perm),
                                  svmin_f64_z(svptrue_b64(), input5, input5Perm));

                const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input7Perm = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));

                input6 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input6, input6Perm),
                                  svmin_f64_z(svptrue_b64(), input6, input6Perm));


                input7 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input7, input7Perm),
                                  svmin_f64_z(svptrue_b64(), input7, input7Perm));
            }
        }
    }
    {
        CoreExchange(input, input2);
        CoreExchange(input3, input4);
    }
    {
        CoreExchange(input5, input6);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, input3Perm),
                              svmin_f64_z(svptrue_b64(), input3, input3Perm));


            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, input4Perm),
                              svmin_f64_z(svptrue_b64(), input4, input4Perm));

            const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, input5Perm),
                              svmin_f64_z(svptrue_b64(), input5, input5Perm));


            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, input6Perm),
                              svmin_f64_z(svptrue_b64(), input6, input6Perm));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, input3Perm),
                              svmin_f64_z(svptrue_b64(), input3, input3Perm));


            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, input4Perm),
                              svmin_f64_z(svptrue_b64(), input4, input4Perm));

            const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, input5Perm),
                              svmin_f64_z(svptrue_b64(), input5, input5Perm));


            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, input6Perm),
                              svmin_f64_z(svptrue_b64(), input6, input6Perm));
        }
    }
    {
        CoreExchange(input, input4);
        CoreExchange(input2, input3);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchange(input6, input7);
    }
    {
        CoreExchangeEnd(input5, input6);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, input5Perm),
                              svmin_f64_z(svptrue_b64(), input5, input5Perm));

            const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input7Perm = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, input6Perm),
                              svmin_f64_z(svptrue_b64(), input6, input6Perm));


            input7 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input7, input7Perm),
                              svmin_f64_z(svptrue_b64(), input7, input7Perm));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, input5Perm),
                              svmin_f64_z(svptrue_b64(), input5, input5Perm));

            const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input7Perm = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, input6Perm),
                              svmin_f64_z(svptrue_b64(), input6, input6Perm));

            input7 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input7, input7Perm),
                              svmin_f64_z(svptrue_b64(), input7, input7Perm));
        }
    }
#endif
    {
        CoreExchange(input2, input7);
        CoreExchange(input3, input6);
        CoreExchange(input4, input5);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4);
    CoreSmallEnd3(input5, input6, input7);
#else
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchangeEnd(input5, input7);
    }
    {
        CoreExchangeEnd(input5, input6);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm5 = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm6 = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm7 = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, inputPerm5),
                              svmin_f64_z(svptrue_b64(), input5, inputPerm5));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, inputPerm6),
                              svmin_f64_z(svptrue_b64(), input6, inputPerm6));

            input7 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input7, inputPerm7),
                              svmin_f64_z(svptrue_b64(), input7, inputPerm7));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm5 = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm6 = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm7 = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, inputPerm5),
                              svmin_f64_z(svptrue_b64(), input5, inputPerm5));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, inputPerm6),
                              svmin_f64_z(svptrue_b64(), input6, inputPerm6));

            input7 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input7, inputPerm7),
                              svmin_f64_z(svptrue_b64(), input7, inputPerm7));
        }
    }
#endif
}


inline void CoreSmallEnd8(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                          svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8){
    {
        CoreExchangeEnd(input, input5);
        CoreExchangeEnd(input2, input6);
        CoreExchangeEnd(input3, input7);
        CoreExchangeEnd(input4, input8);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4);
    CoreSmallEnd4(input5, input6, input7, input8);
#else
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchangeEnd(input5, input7);
        CoreExchangeEnd(input6, input8);
    }
    {
        CoreExchangeEnd(input5, input6);
        CoreExchangeEnd(input7, input8);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm5 = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm6 = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm7 = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm8 = svtbl_f64(input8, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, inputPerm5),
                              svmin_f64_z(svptrue_b64(), input5, inputPerm5));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, inputPerm6),
                              svmin_f64_z(svptrue_b64(), input6, inputPerm6));

            input7 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input7, inputPerm7),
                              svmin_f64_z(svptrue_b64(), input7, inputPerm7));

            input8 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input8, inputPerm8),
                              svmin_f64_z(svptrue_b64(), input8, inputPerm8));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm5 = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm6 = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm7 = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm8 = svtbl_f64(input8, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, inputPerm5),
                              svmin_f64_z(svptrue_b64(), input5, inputPerm5));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, inputPerm6),
                              svmin_f64_z(svptrue_b64(), input6, inputPerm6));

            input7 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input7, inputPerm7),
                              svmin_f64_z(svptrue_b64(), input7, inputPerm7));

            input8 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input8, inputPerm8),
                              svmin_f64_z(svptrue_b64(), input8, inputPerm8));
        }
    }
#endif
}

inline void CoreSmallSort8(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                           svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8){
#ifdef NOOPTIM
    CoreSmallSort4(input, input2, input3, input4);
    CoreSmallSort4(input5, input6, input7, input8);
#else
    {
        const int nbValuesInVec = svcntd();

        const svint64_t vecindex = svindex_s64(0, 1);

        svbool_t ffttout = svzip1_b64(svpfalse_b(),svptrue_b64());

        svint64_t vecincout = svsel_s64(ffttout, svdup_s64(-1), svdup_s64(1));

        {// stepout == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, vecincout);

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));


            input2 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input2, input2Perm),
                              svmin_f64_z(svptrue_b64(), input2, input2Perm));

            const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input3 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input3, input3Perm),
                              svmin_f64_z(svptrue_b64(), input3, input3Perm));


            input4 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input4, input4Perm),
                              svmin_f64_z(svptrue_b64(), input4, input4Perm));

            const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

            input5 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input5, input5Perm),
                              svmin_f64_z(svptrue_b64(), input5, input5Perm));


            input6 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input6, input6Perm),
                              svmin_f64_z(svptrue_b64(), input6, input6Perm));

            const svfloat64_t input7Perm = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input8Perm = svtbl_f64(input8, svreinterpret_u64_s64(vecpermute));

            input7 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input7, input7Perm),
                              svmin_f64_z(svptrue_b64(), input7, input7Perm));


            input8 = svsel_f64(ffttout,
                              svmax_f64_z(svptrue_b64(), input8, input8Perm),
                              svmin_f64_z(svptrue_b64(), input8, input8Perm));
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b64(ffttout,ffttout);
            vecincout = svsel_s64(ffttout,
                                      svsub_n_s64_z(svptrue_b64(), vecincout, stepout),
                                      svadd_n_s64_z(svptrue_b64(), vecincout, stepout));
            {
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, vecincout);

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));


                input2 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));

                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

                input3 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));


                input4 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input4, input4Perm),
                                  svmin_f64_z(svptrue_b64(), input4, input4Perm));

                const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

                input5 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input5, input5Perm),
                                  svmin_f64_z(svptrue_b64(), input5, input5Perm));


                input6 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input6, input6Perm),
                                  svmin_f64_z(svptrue_b64(), input6, input6Perm));

                const svfloat64_t input7Perm = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input8Perm = svtbl_f64(input8, svreinterpret_u64_s64(vecpermute));

                input7 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input7, input7Perm),
                                  svmin_f64_z(svptrue_b64(), input7, input7Perm));


                input8 = svsel_f64(ffttout,
                                  svmax_f64_z(svptrue_b64(), input8, input8Perm),
                                  svmin_f64_z(svptrue_b64(), input8, input8Perm));
            }

            svbool_t fftt = svuzp2_b64(ffttout,ffttout);

            svint64_t vecinc = svdup_s64(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));

                input2 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));

                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

                input3 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));

                input4 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input4, input4Perm),
                                  svmin_f64_z(svptrue_b64(), input4, input4Perm));

                const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

                input5 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input5, input5Perm),
                                  svmin_f64_z(svptrue_b64(), input5, input5Perm));

                input6 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input6, input6Perm),
                                  svmin_f64_z(svptrue_b64(), input6, input6Perm));

                const svfloat64_t input7Perm = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input8Perm = svtbl_f64(input8, svreinterpret_u64_s64(vecpermute));

                input7 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input7, input7Perm),
                                  svmin_f64_z(svptrue_b64(), input7, input7Perm));

                input8 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input8, input8Perm),
                                  svmin_f64_z(svptrue_b64(), input8, input8Perm));

                fftt = svuzp2_b64(fftt,fftt);
                vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

                const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input2Perm = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

                input = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input, inputPerm),
                                  svmin_f64_z(svptrue_b64(), input, inputPerm));

                input2 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input2, input2Perm),
                                  svmin_f64_z(svptrue_b64(), input2, input2Perm));

                const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

                input3 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input3, input3Perm),
                                  svmin_f64_z(svptrue_b64(), input3, input3Perm));

                input4 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input4, input4Perm),
                                  svmin_f64_z(svptrue_b64(), input4, input4Perm));

                const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));

                input5 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input5, input5Perm),
                                  svmin_f64_z(svptrue_b64(), input5, input5Perm));

                input6 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input6, input6Perm),
                                  svmin_f64_z(svptrue_b64(), input6, input6Perm));

                const svfloat64_t input7Perm = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));
                const svfloat64_t input8Perm = svtbl_f64(input8, svreinterpret_u64_s64(vecpermute));

                input7 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input7, input7Perm),
                                  svmin_f64_z(svptrue_b64(), input7, input7Perm));

                input8 = svsel_f64(fftt,
                                  svmax_f64_z(svptrue_b64(), input8, input8Perm),
                                  svmin_f64_z(svptrue_b64(), input8, input8Perm));
            }
        }
    }
    {
        CoreExchange(input, input2);
        CoreExchange(input3, input4);
    }
    {
        CoreExchange(input5, input6);
        CoreExchange(input7, input8);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, input3Perm),
                              svmin_f64_z(svptrue_b64(), input3, input3Perm));


            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, input4Perm),
                              svmin_f64_z(svptrue_b64(), input4, input4Perm));

            const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input7Perm = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input8Perm = svtbl_f64(input8, svreinterpret_u64_s64(vecpermute));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, input5Perm),
                              svmin_f64_z(svptrue_b64(), input5, input5Perm));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, input6Perm),
                              svmin_f64_z(svptrue_b64(), input6, input6Perm));

            input7 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input7, input7Perm),
                              svmin_f64_z(svptrue_b64(), input7, input7Perm));

            input8 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input8, input8Perm),
                              svmin_f64_z(svptrue_b64(), input8, input8Perm));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            const svfloat64_t input3Perm = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input4Perm = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, input3Perm),
                              svmin_f64_z(svptrue_b64(), input3, input3Perm));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, input4Perm),
                              svmin_f64_z(svptrue_b64(), input4, input4Perm));

            const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input7Perm = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input8Perm = svtbl_f64(input8, svreinterpret_u64_s64(vecpermute));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, input5Perm),
                              svmin_f64_z(svptrue_b64(), input5, input5Perm));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, input6Perm),
                              svmin_f64_z(svptrue_b64(), input6, input6Perm));

            input7 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input7, input7Perm),
                              svmin_f64_z(svptrue_b64(), input7, input7Perm));

            input8 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input8, input8Perm),
                              svmin_f64_z(svptrue_b64(), input8, input8Perm));
        }
    }
    {
        CoreExchange(input, input4);
        CoreExchange(input2, input3);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchange(input5, input8);
        CoreExchange(input6, input7);
    }
    {
        CoreExchangeEnd(input5, input6);
        CoreExchangeEnd(input7, input8);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input7Perm = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input8Perm = svtbl_f64(input8, svreinterpret_u64_s64(vecpermute));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, input5Perm),
                              svmin_f64_z(svptrue_b64(), input5, input5Perm));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, input6Perm),
                              svmin_f64_z(svptrue_b64(), input6, input6Perm));

            input7 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input7, input7Perm),
                              svmin_f64_z(svptrue_b64(), input7, input7Perm));

            input8 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input8, input8Perm),
                              svmin_f64_z(svptrue_b64(), input8, input8Perm));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            const svfloat64_t input5Perm = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input6Perm = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input7Perm = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t input8Perm = svtbl_f64(input8, svreinterpret_u64_s64(vecpermute));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, input5Perm),
                              svmin_f64_z(svptrue_b64(), input5, input5Perm));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, input6Perm),
                              svmin_f64_z(svptrue_b64(), input6, input6Perm));

            input7 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input7, input7Perm),
                              svmin_f64_z(svptrue_b64(), input7, input7Perm));

            input8 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input8, input8Perm),
                              svmin_f64_z(svptrue_b64(), input8, input8Perm));
        }
    }
#endif
    {
        CoreExchange(input, input8);
        CoreExchange(input2, input7);
        CoreExchange(input3, input6);
        CoreExchange(input4, input5);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4);
    CoreSmallEnd4(input5, input6, input7, input8);
#else
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchangeEnd(input5, input7);
        CoreExchangeEnd(input6, input8);
    }
    {
        CoreExchangeEnd(input5, input6);
        CoreExchangeEnd(input7, input8);
    }
    {
        const int nbValuesInVec = svcntd();

        svbool_t fftt = getFalseTrueMask64(nbValuesInVec/2);

        const svint64_t vecindex = svindex_s64(0, 1);

        svint64_t vecinc = svdup_s64(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm5 = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm6 = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm7 = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm8 = svtbl_f64(input8, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, inputPerm5),
                              svmin_f64_z(svptrue_b64(), input5, inputPerm5));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, inputPerm6),
                              svmin_f64_z(svptrue_b64(), input6, inputPerm6));

            input7 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input7, inputPerm7),
                              svmin_f64_z(svptrue_b64(), input7, inputPerm7));

            input8 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input8, inputPerm8),
                              svmin_f64_z(svptrue_b64(), input8, inputPerm8));

            fftt = svuzp2_b64(fftt,fftt);
            vecinc = svdiv_n_s64_z(svptrue_b64(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint64_t vecpermute = svadd_s64_z(svptrue_b64(), vecindex, svsel_s64(fftt, svneg_s64_z(fftt, vecinc), vecinc));

            const svfloat64_t inputPerm = svtbl_f64(input, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm2 = svtbl_f64(input2, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm3 = svtbl_f64(input3, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm4 = svtbl_f64(input4, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm5 = svtbl_f64(input5, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm6 = svtbl_f64(input6, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm7 = svtbl_f64(input7, svreinterpret_u64_s64(vecpermute));
            const svfloat64_t inputPerm8 = svtbl_f64(input8, svreinterpret_u64_s64(vecpermute));

            input = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input, inputPerm),
                              svmin_f64_z(svptrue_b64(), input, inputPerm));

            input2 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input2, inputPerm2),
                              svmin_f64_z(svptrue_b64(), input2, inputPerm2));

            input3 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input3, inputPerm3),
                              svmin_f64_z(svptrue_b64(), input3, inputPerm3));

            input4 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input4, inputPerm4),
                              svmin_f64_z(svptrue_b64(), input4, inputPerm4));

            input5 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input5, inputPerm5),
                              svmin_f64_z(svptrue_b64(), input5, inputPerm5));

            input6 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input6, inputPerm6),
                              svmin_f64_z(svptrue_b64(), input6, inputPerm6));

            input7 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input7, inputPerm7),
                              svmin_f64_z(svptrue_b64(), input7, inputPerm7));

            input8 = svsel_f64(fftt,
                              svmax_f64_z(svptrue_b64(), input8, inputPerm8),
                              svmin_f64_z(svptrue_b64(), input8, inputPerm8));
        }
    }
#endif
}

inline void CoreSmallSort9(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                           svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8,
                           svfloat64_t& input9){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort(input9);
    {
        CoreExchange(input8, input9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSortEnd(input9);
}

inline void CoreSmallSort10(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                            svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8,
                            svfloat64_t& input9, svfloat64_t& input10){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort2(input9, input10);
    {
        CoreExchange(input7, input10);
        CoreExchange(input8, input9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd2(input9, input10);
}

inline void CoreSmallSort11(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                            svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8,
                            svfloat64_t& input9, svfloat64_t& input10, svfloat64_t& input11){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort3(input9, input10, input11);
    {
        CoreExchange(input6, input11);
        CoreExchange(input7, input10);
        CoreExchange(input8, input9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd3(input9, input10, input11);
}

inline void CoreSmallSort12(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                            svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8,
                            svfloat64_t& input9, svfloat64_t& input10, svfloat64_t& input11, svfloat64_t& input12){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort4(input9, input10, input11, input12);
    {
        CoreExchange(input5, input12);
        CoreExchange(input6, input11);
        CoreExchange(input7, input10);
        CoreExchange(input8, input9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd4(input9, input10, input11, input12);
}

inline void CoreSmallSort13(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                            svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8,
                            svfloat64_t& input9, svfloat64_t& input10, svfloat64_t& input11, svfloat64_t& input12,
                            svfloat64_t& input13){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort5(input9, input10, input11, input12, input13);
    {
        CoreExchange(input4, input13);
        CoreExchange(input5, input12);
        CoreExchange(input6, input11);
        CoreExchange(input7, input10);
        CoreExchange(input8, input9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd5(input9, input10, input11, input12, input13);
}

inline void CoreSmallSort14(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                            svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8,
                            svfloat64_t& input9, svfloat64_t& input10, svfloat64_t& input11, svfloat64_t& input12,
                            svfloat64_t& input13, svfloat64_t& input14){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort6(input9, input10, input11, input12, input13, input14);
    {
        CoreExchange(input3, input14);
        CoreExchange(input4, input13);
        CoreExchange(input5, input12);
        CoreExchange(input6, input11);
        CoreExchange(input7, input10);
        CoreExchange(input8, input9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd6(input9, input10, input11, input12, input13, input14);
}

inline void CoreSmallSort15(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                            svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8,
                            svfloat64_t& input9, svfloat64_t& input10, svfloat64_t& input11, svfloat64_t& input12,
                            svfloat64_t& input13, svfloat64_t& input14, svfloat64_t& input15){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort7(input9, input10, input11, input12, input13, input14, input15);
    {
        CoreExchange(input2, input15);
        CoreExchange(input3, input14);
        CoreExchange(input4, input13);
        CoreExchange(input5, input12);
        CoreExchange(input6, input11);
        CoreExchange(input7, input10);
        CoreExchange(input8, input9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd7(input9, input10, input11, input12, input13, input14, input15);
}

inline void CoreSmallSort16(svfloat64_t& input, svfloat64_t& input2, svfloat64_t& input3, svfloat64_t& input4,
                            svfloat64_t& input5, svfloat64_t& input6, svfloat64_t& input7, svfloat64_t& input8,
                            svfloat64_t& input9, svfloat64_t& input10, svfloat64_t& input11, svfloat64_t& input12,
                            svfloat64_t& input13, svfloat64_t& input14, svfloat64_t& input15, svfloat64_t& input16){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort8(input9, input10, input11, input12, input13, input14, input15, input16);
    {
        CoreExchange(input, input16);
        CoreExchange(input2, input15);
        CoreExchange(input3, input14);
        CoreExchange(input4, input13);
        CoreExchange(input5, input12);
        CoreExchange(input6, input11);
        CoreExchange(input7, input10);
        CoreExchange(input8, input9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd8(input9, input10, input11, input12, input13, input14, input15, input16);
}

inline void CoreSmallEnd2(svint32_t& input, svint32_t& input2){
    if(svmaxv_s32(svptrue_b32(),input) > svminv_s32(svptrue_b32(),input2)){
        CoreExchangeEnd(input, input2);
    }
#ifdef NOOPTIM
    CoreSmallSortEnd(input);
    CoreSmallSortEnd(input2);
#else
    bool isSorted1 = IsSorted(input);
    bool isSorted2 = IsSorted(input2);
    if(!isSorted1 && !isSorted2){
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
            isSorted1 |= IsSorted(input);
            isSorted2 |= IsSorted(input2);
            if(isSorted1 && isSorted2){
                return;
            }
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
        }
    }
    else if(!isSorted1){
        CoreSmallSortEnd(input);
    }
    else if(!isSorted2){
        CoreSmallSortEnd(input2);
    }
#endif
}

inline void CoreSmallSort2(svint32_t& input, svint32_t& input2){
#ifdef NOOPTIM
    CoreSmallSort(input);
    CoreSmallSort(input2);
#else
    bool isSorted1 = IsSorted(input);
    bool isSorted2 = IsSorted(input2);
    if(!isSorted1 && !isSorted2){
        const int nbValuesInVec = svcntw();

        const svint32_t vecindex = svindex_s32(0, 1);

        svbool_t ffttout = svzip1_b32(svpfalse_b(),svptrue_b32());

        svint32_t vecincout = svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

        {// stepout == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));

            input2 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input2, input2Perm),
                              svmin_s32_z(svptrue_b32(), input2, input2Perm));
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec && !(isSorted1 && isSorted2) ; stepout *= 2){
            ffttout = svzip1_b32(ffttout,ffttout);
            vecincout = svsel_s32(ffttout,
                                      svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                                      svadd_n_s32_z(svptrue_b32(), vecincout, stepout));
            {
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
            }

            svbool_t fftt = svuzp2_b32(ffttout,ffttout);

            svint32_t vecinc = svdup_s32(stepout/2);

            for(long int step = stepout/2 ; step > 1 && !(isSorted1 && isSorted2) ; step/=2){
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));

                fftt = svuzp2_b32(fftt,fftt);
                vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
                isSorted1 |= IsSorted(input);
                isSorted2 |= IsSorted(input2);
            }
            { // Step == 1
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
            }
        }
    }
    else if(!isSorted1){
        CoreSmallSort(input);
    }
    else if(!isSorted2){
        CoreSmallSort(input2);
    }
#endif
    if(svmaxv_s32(svptrue_b32(),input) > svminv_s32(svptrue_b32(),input2)){
        CoreExchange(input, input2);
    }
#ifdef NOOPTIM
    CoreSmallSortEnd(input);
    CoreSmallSortEnd(input2);
#else
    isSorted1 = IsSorted(input);
    isSorted2 = IsSorted(input2);
    if(!isSorted1 && !isSorted2){
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
            isSorted1 |= IsSorted(input);
            isSorted2 |= IsSorted(input2);
            if(isSorted1 && isSorted2){
                return;
            }
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
        }
    }
    else if(!isSorted1){
        CoreSmallSortEnd(input);
    }
    else if(!isSorted2){
        CoreSmallSortEnd(input2);
    }
#endif
}

inline void CoreSmallEnd3(svint32_t& input, svint32_t& input2, svint32_t& input3){
    {
        CoreExchangeEnd(input, input3);
    }
#ifdef NOOPTIM
    CoreSmallEnd2(input, input2);
    CoreSmallSortEnd(input3);
#else
    {
        CoreExchangeEnd(input, input2);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
        }
    }
#endif
}

inline void CoreSmallSort3(svint32_t& input, svint32_t& input2, svint32_t& input3){
#ifdef NOOPTIM
    CoreSmallSort2(input, input2);
    CoreSmallSort(input3);
#else
    {
        const int nbValuesInVec = svcntw();

        const svint32_t vecindex = svindex_s32(0, 1);

        svbool_t ffttout = svzip1_b32(svpfalse_b(),svptrue_b32());

        svint32_t vecincout = svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

        {// stepout == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));

            input2 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input2, input2Perm),
                              svmin_s32_z(svptrue_b32(), input2, input2Perm));
            input3 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input3, input3Perm),
                              svmin_s32_z(svptrue_b32(), input3, input3Perm));
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b32(ffttout,ffttout);
            vecincout = svsel_s32(ffttout,
                                      svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                                      svadd_n_s32_z(svptrue_b32(), vecincout, stepout));
            {
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
            }

            svbool_t fftt = svuzp2_b32(ffttout,ffttout);

            svint32_t vecinc = svdup_s32(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));

                fftt = svuzp2_b32(fftt,fftt);
                vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
            }
        }
    }
    {
        CoreExchange(input, input2);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
        }
    }
#endif
    {
        CoreExchange(input2, input3);
    }
#ifdef NOOPTIM
    CoreSmallEnd2(input, input2);
    CoreSmallSortEnd(input3);
#else
    {
        CoreExchangeEnd(input, input2);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
        }
    }
#endif
}

inline void CoreSmallEnd4(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4){
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
#ifdef NOOPTIM
    CoreSmallEnd2(input, input2);
    CoreSmallEnd2(input3, input4);
#else
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
        }
    }
#endif
}

inline void CoreSmallSort4(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4){
#ifdef NOOPTIM
    CoreSmallSort2(input, input2);
    CoreSmallSort2(input3, input4);
#else
    {
        const int nbValuesInVec = svcntw();

        const svint32_t vecindex = svindex_s32(0, 1);

        svbool_t ffttout = svzip1_b32(svpfalse_b(),svptrue_b32());

        svint32_t vecincout = svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

        {// stepout == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));

            input2 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input2, input2Perm),
                              svmin_s32_z(svptrue_b32(), input2, input2Perm));
            input3 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input3, input3Perm),
                              svmin_s32_z(svptrue_b32(), input3, input3Perm));
            input4 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input4, input4Perm),
                              svmin_s32_z(svptrue_b32(), input4, input4Perm));
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b32(ffttout,ffttout);
            vecincout = svsel_s32(ffttout,
                                      svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                                      svadd_n_s32_z(svptrue_b32(), vecincout, stepout));
            {
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
                input4 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input4, input4Perm),
                                  svmin_s32_z(svptrue_b32(), input4, input4Perm));
            }

            svbool_t fftt = svuzp2_b32(ffttout,ffttout);

            svint32_t vecinc = svdup_s32(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
                input4 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input4, input4Perm),
                                  svmin_s32_z(svptrue_b32(), input4, input4Perm));

                fftt = svuzp2_b32(fftt,fftt);
                vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
                input4 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input4, input4Perm),
                                  svmin_s32_z(svptrue_b32(), input4, input4Perm));
            }
        }
    }
    {
        CoreExchange(input, input2);
    }
    {
        CoreExchange(input3, input4);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, input3Perm),
                              svmin_s32_z(svptrue_b32(), input3, input3Perm));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, input4Perm),
                              svmin_s32_z(svptrue_b32(), input4, input4Perm));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, input3Perm),
                              svmin_s32_z(svptrue_b32(), input3, input3Perm));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, input4Perm),
                              svmin_s32_z(svptrue_b32(), input4, input4Perm));
        }
    }
#endif
    {
        CoreExchange(input, input4);
        CoreExchange(input2, input3);
    }
#ifdef NOOPTIM
    CoreSmallEnd2(input, input2);
    CoreSmallEnd2(input3, input4);
#else
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
        }
    }
#endif
}

inline void CoreSmallEnd5(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4, svint32_t& input5){
    {
        CoreExchangeEnd(input, input5);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4);
    CoreSmallSortEnd(input5);
#else
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, inputPerm5),
                              svmin_s32_z(svptrue_b32(), input5, inputPerm5));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, inputPerm5),
                              svmin_s32_z(svptrue_b32(), input5, inputPerm5));
        }
    }
#endif
}

inline void CoreSmallSort5(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                           svint32_t& input5){
#ifdef NOOPTIM
    CoreSmallSort4(input, input2, input3, input4);
    CoreSmallSort(input5);
#else
    {
        const int nbValuesInVec = svcntw();

        const svint32_t vecindex = svindex_s32(0, 1);

        svbool_t ffttout = svzip1_b32(svpfalse_b(),svptrue_b32());

        svint32_t vecincout = svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

        {// stepout == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));

            input2 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input2, input2Perm),
                              svmin_s32_z(svptrue_b32(), input2, input2Perm));
            input3 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input3, input3Perm),
                              svmin_s32_z(svptrue_b32(), input3, input3Perm));
            input4 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input4, input4Perm),
                              svmin_s32_z(svptrue_b32(), input4, input4Perm));

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            input5 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input5, input5Perm),
                              svmin_s32_z(svptrue_b32(), input5, input5Perm));
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b32(ffttout,ffttout);
            vecincout = svsel_s32(ffttout,
                                      svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                                      svadd_n_s32_z(svptrue_b32(), vecincout, stepout));
            {
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
                input4 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input4, input4Perm),
                                  svmin_s32_z(svptrue_b32(), input4, input4Perm));

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                input5 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input5, input5Perm),
                                  svmin_s32_z(svptrue_b32(), input5, input5Perm));
            }

            svbool_t fftt = svuzp2_b32(ffttout,ffttout);

            svint32_t vecinc = svdup_s32(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
                input4 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input4, input4Perm),
                                  svmin_s32_z(svptrue_b32(), input4, input4Perm));

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                input5 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input5, input5Perm),
                                  svmin_s32_z(svptrue_b32(), input5, input5Perm));

                fftt = svuzp2_b32(fftt,fftt);
                vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
                input4 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input4, input4Perm),
                                  svmin_s32_z(svptrue_b32(), input4, input4Perm));

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                input5 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input5, input5Perm),
                                  svmin_s32_z(svptrue_b32(), input5, input5Perm));
            }
        }
    }
    {
        CoreExchange(input, input2);
    }
    {
        CoreExchange(input3, input4);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, input3Perm),
                              svmin_s32_z(svptrue_b32(), input3, input3Perm));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, input4Perm),
                              svmin_s32_z(svptrue_b32(), input4, input4Perm));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, input3Perm),
                              svmin_s32_z(svptrue_b32(), input3, input3Perm));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, input4Perm),
                              svmin_s32_z(svptrue_b32(), input4, input4Perm));
        }
    }
    {
        CoreExchange(input, input4);
        CoreExchange(input2, input3);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
        }
    }
#endif
    {
        CoreExchange(input4, input5);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4);
    CoreSmallSortEnd(input5);
#else
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, inputPerm5),
                              svmin_s32_z(svptrue_b32(), input5, inputPerm5));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, inputPerm5),
                              svmin_s32_z(svptrue_b32(), input5, inputPerm5));
        }
    }
#endif
}

inline void CoreSmallEnd6(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                          svint32_t& input5, svint32_t& input6){
    {
        CoreExchangeEnd(input, input5);
        CoreExchangeEnd(input2, input6);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4);
    CoreSmallEnd2(input5, input6);
#else
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchangeEnd(input5, input6);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, inputPerm5),
                              svmin_s32_z(svptrue_b32(), input5, inputPerm5));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, inputPerm6),
                              svmin_s32_z(svptrue_b32(), input6, inputPerm6));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, inputPerm5),
                              svmin_s32_z(svptrue_b32(), input5, inputPerm5));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, inputPerm6),
                              svmin_s32_z(svptrue_b32(), input6, inputPerm6));
        }
    }
#endif
}

inline void CoreSmallSort6(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                           svint32_t& input5, svint32_t& input6){
#ifdef NOOPTIM
    CoreSmallSort4(input, input2, input3, input4);
    CoreSmallSort2(input5, input6);
#else
    {
        const int nbValuesInVec = svcntw();

        const svint32_t vecindex = svindex_s32(0, 1);

        svbool_t ffttout = svzip1_b32(svpfalse_b(),svptrue_b32());

        svint32_t vecincout = svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

        {// stepout == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));

            input2 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input2, input2Perm),
                              svmin_s32_z(svptrue_b32(), input2, input2Perm));
            input3 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input3, input3Perm),
                              svmin_s32_z(svptrue_b32(), input3, input3Perm));
            input4 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input4, input4Perm),
                              svmin_s32_z(svptrue_b32(), input4, input4Perm));

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            input5 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input5, input5Perm),
                              svmin_s32_z(svptrue_b32(), input5, input5Perm));
            input6 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input6, input6Perm),
                              svmin_s32_z(svptrue_b32(), input6, input6Perm));
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b32(ffttout,ffttout);
            vecincout = svsel_s32(ffttout,
                                      svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                                      svadd_n_s32_z(svptrue_b32(), vecincout, stepout));
            {
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
                input4 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input4, input4Perm),
                                  svmin_s32_z(svptrue_b32(), input4, input4Perm));

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                input5 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input5, input5Perm),
                                  svmin_s32_z(svptrue_b32(), input5, input5Perm));
                input6 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input6, input6Perm),
                                  svmin_s32_z(svptrue_b32(), input6, input6Perm));
            }

            svbool_t fftt = svuzp2_b32(ffttout,ffttout);

            svint32_t vecinc = svdup_s32(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
                input4 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input4, input4Perm),
                                  svmin_s32_z(svptrue_b32(), input4, input4Perm));

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                input5 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input5, input5Perm),
                                  svmin_s32_z(svptrue_b32(), input5, input5Perm));
                input6 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input6, input6Perm),
                                  svmin_s32_z(svptrue_b32(), input6, input6Perm));

                fftt = svuzp2_b32(fftt,fftt);
                vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
                input4 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input4, input4Perm),
                                  svmin_s32_z(svptrue_b32(), input4, input4Perm));

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                input5 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input5, input5Perm),
                                  svmin_s32_z(svptrue_b32(), input5, input5Perm));
                input6 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input6, input6Perm),
                                  svmin_s32_z(svptrue_b32(), input6, input6Perm));
            }
        }
    }
    {
        CoreExchange(input, input2);
    }
    {
        CoreExchange(input3, input4);
    }
    {
        CoreExchange(input5, input6);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, input3Perm),
                              svmin_s32_z(svptrue_b32(), input3, input3Perm));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, input4Perm),
                              svmin_s32_z(svptrue_b32(), input4, input4Perm));

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, input5Perm),
                              svmin_s32_z(svptrue_b32(), input5, input5Perm));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, input6Perm),
                              svmin_s32_z(svptrue_b32(), input6, input6Perm));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, input3Perm),
                              svmin_s32_z(svptrue_b32(), input3, input3Perm));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, input4Perm),
                              svmin_s32_z(svptrue_b32(), input4, input4Perm));

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, input5Perm),
                              svmin_s32_z(svptrue_b32(), input5, input5Perm));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, input6Perm),
                              svmin_s32_z(svptrue_b32(), input6, input6Perm));
        }
    }
    {
        CoreExchange(input, input4);
        CoreExchange(input2, input3);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchangeEnd(input5, input6);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, input5Perm),
                              svmin_s32_z(svptrue_b32(), input5, input5Perm));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, input6Perm),
                              svmin_s32_z(svptrue_b32(), input6, input6Perm));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, input5Perm),
                              svmin_s32_z(svptrue_b32(), input5, input5Perm));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, input6Perm),
                              svmin_s32_z(svptrue_b32(), input6, input6Perm));
        }
    }
#endif
    {
        CoreExchange(input3, input6);
        CoreExchange(input4, input5);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4);
    CoreSmallEnd2(input5, input6);
#else
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchangeEnd(input5, input6);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, inputPerm5),
                              svmin_s32_z(svptrue_b32(), input5, inputPerm5));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, inputPerm6),
                              svmin_s32_z(svptrue_b32(), input6, inputPerm6));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, inputPerm5),
                              svmin_s32_z(svptrue_b32(), input5, inputPerm5));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, inputPerm6),
                              svmin_s32_z(svptrue_b32(), input6, inputPerm6));
        }
    }
#endif
}

inline void CoreSmallEnd7(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                          svint32_t& input5, svint32_t& input6, svint32_t& input7){
    {
        CoreExchangeEnd(input, input5);
        CoreExchangeEnd(input2, input6);
        CoreExchangeEnd(input3, input7);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4);
    CoreSmallEnd3(input5, input6, input7);
#else
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchangeEnd(input5, input7);
    }
    {
        CoreExchangeEnd(input5, input6);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7 = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, inputPerm5),
                              svmin_s32_z(svptrue_b32(), input5, inputPerm5));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, inputPerm6),
                              svmin_s32_z(svptrue_b32(), input6, inputPerm6));
            input7 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input7, inputPerm7),
                              svmin_s32_z(svptrue_b32(), input7, inputPerm7));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7 = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, inputPerm5),
                              svmin_s32_z(svptrue_b32(), input5, inputPerm5));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, inputPerm6),
                              svmin_s32_z(svptrue_b32(), input6, inputPerm6));
            input7 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input7, inputPerm7),
                              svmin_s32_z(svptrue_b32(), input7, inputPerm7));
        }
    }
#endif
}

inline void CoreSmallSort7(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                           svint32_t& input5, svint32_t& input6, svint32_t& input7){
#ifdef NOOPTIM
    CoreSmallSort4(input, input2, input3, input4);
    CoreSmallSort3(input5, input6, input7);
#else
    {
        const int nbValuesInVec = svcntw();

        const svint32_t vecindex = svindex_s32(0, 1);

        svbool_t ffttout = svzip1_b32(svpfalse_b(),svptrue_b32());

        svint32_t vecincout = svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

        {// stepout == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));

            input2 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input2, input2Perm),
                              svmin_s32_z(svptrue_b32(), input2, input2Perm));
            input3 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input3, input3Perm),
                              svmin_s32_z(svptrue_b32(), input3, input3Perm));
            input4 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input4, input4Perm),
                              svmin_s32_z(svptrue_b32(), input4, input4Perm));

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            input5 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input5, input5Perm),
                              svmin_s32_z(svptrue_b32(), input5, input5Perm));
            input6 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input6, input6Perm),
                              svmin_s32_z(svptrue_b32(), input6, input6Perm));
            input7 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input7, input7Perm),
                              svmin_s32_z(svptrue_b32(), input7, input7Perm));
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b32(ffttout,ffttout);
            vecincout = svsel_s32(ffttout,
                                      svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                                      svadd_n_s32_z(svptrue_b32(), vecincout, stepout));
            {
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
                input4 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input4, input4Perm),
                                  svmin_s32_z(svptrue_b32(), input4, input4Perm));

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
                input5 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input5, input5Perm),
                                  svmin_s32_z(svptrue_b32(), input5, input5Perm));
                input6 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input6, input6Perm),
                                  svmin_s32_z(svptrue_b32(), input6, input6Perm));
                input7 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input7, input7Perm),
                                  svmin_s32_z(svptrue_b32(), input7, input7Perm));
            }

            svbool_t fftt = svuzp2_b32(ffttout,ffttout);

            svint32_t vecinc = svdup_s32(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
                input4 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input4, input4Perm),
                                  svmin_s32_z(svptrue_b32(), input4, input4Perm));

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
                input5 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input5, input5Perm),
                                  svmin_s32_z(svptrue_b32(), input5, input5Perm));
                input6 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input6, input6Perm),
                                  svmin_s32_z(svptrue_b32(), input6, input6Perm));
                input7 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input7, input7Perm),
                                  svmin_s32_z(svptrue_b32(), input7, input7Perm));

                fftt = svuzp2_b32(fftt,fftt);
                vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
                input4 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input4, input4Perm),
                                  svmin_s32_z(svptrue_b32(), input4, input4Perm));

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
                input5 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input5, input5Perm),
                                  svmin_s32_z(svptrue_b32(), input5, input5Perm));
                input6 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input6, input6Perm),
                                  svmin_s32_z(svptrue_b32(), input6, input6Perm));
                input7 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input7, input7Perm),
                                  svmin_s32_z(svptrue_b32(), input7, input7Perm));
            }
        }
    }
    {
        CoreExchange(input, input2);
    }
    {
        CoreExchange(input3, input4);
    }
    {
        CoreExchange(input5, input6);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, input3Perm),
                              svmin_s32_z(svptrue_b32(), input3, input3Perm));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, input4Perm),
                              svmin_s32_z(svptrue_b32(), input4, input4Perm));

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, input5Perm),
                              svmin_s32_z(svptrue_b32(), input5, input5Perm));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, input6Perm),
                              svmin_s32_z(svptrue_b32(), input6, input6Perm));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, input3Perm),
                              svmin_s32_z(svptrue_b32(), input3, input3Perm));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, input4Perm),
                              svmin_s32_z(svptrue_b32(), input4, input4Perm));

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, input5Perm),
                              svmin_s32_z(svptrue_b32(), input5, input5Perm));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, input6Perm),
                              svmin_s32_z(svptrue_b32(), input6, input6Perm));
        }
    }
    {
        CoreExchange(input, input4);
        CoreExchange(input2, input3);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchange(input6, input7);
    }
    {
        CoreExchangeEnd(input5, input6);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, input5Perm),
                              svmin_s32_z(svptrue_b32(), input5, input5Perm));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, input6Perm),
                              svmin_s32_z(svptrue_b32(), input6, input6Perm));
            input7 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input7, input7Perm),
                              svmin_s32_z(svptrue_b32(), input7, input7Perm));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, input5Perm),
                              svmin_s32_z(svptrue_b32(), input5, input5Perm));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, input6Perm),
                              svmin_s32_z(svptrue_b32(), input6, input6Perm));
            input7 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input7, input7Perm),
                              svmin_s32_z(svptrue_b32(), input7, input7Perm));
        }
    }
#endif
    {
        CoreExchange(input2, input7);
        CoreExchange(input3, input6);
        CoreExchange(input4, input5);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4);
    CoreSmallEnd3(input5, input6, input7);
#else
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchangeEnd(input5, input7);
    }
    {
        CoreExchangeEnd(input5, input6);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7 = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, inputPerm5),
                              svmin_s32_z(svptrue_b32(), input5, inputPerm5));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, inputPerm6),
                              svmin_s32_z(svptrue_b32(), input6, inputPerm6));
            input7 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input7, inputPerm7),
                              svmin_s32_z(svptrue_b32(), input7, inputPerm7));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7 = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, inputPerm5),
                              svmin_s32_z(svptrue_b32(), input5, inputPerm5));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, inputPerm6),
                              svmin_s32_z(svptrue_b32(), input6, inputPerm6));
            input7 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input7, inputPerm7),
                              svmin_s32_z(svptrue_b32(), input7, inputPerm7));
        }
    }
#endif
}


inline void CoreSmallEnd8(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                          svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8){
    {
        CoreExchangeEnd(input, input5);
        CoreExchangeEnd(input2, input6);
        CoreExchangeEnd(input3, input7);
        CoreExchangeEnd(input4, input8);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4);
    CoreSmallEnd4(input5, input6, input7, input8);
#else
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchangeEnd(input5, input7);
        CoreExchangeEnd(input6, input8);
    }
    {
        CoreExchangeEnd(input5, input6);
        CoreExchangeEnd(input7, input8);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7 = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm8 = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, inputPerm5),
                              svmin_s32_z(svptrue_b32(), input5, inputPerm5));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, inputPerm6),
                              svmin_s32_z(svptrue_b32(), input6, inputPerm6));
            input7 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input7, inputPerm7),
                              svmin_s32_z(svptrue_b32(), input7, inputPerm7));
            input8 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input8, inputPerm8),
                              svmin_s32_z(svptrue_b32(), input8, inputPerm8));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7 = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm8 = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, inputPerm5),
                              svmin_s32_z(svptrue_b32(), input5, inputPerm5));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, inputPerm6),
                              svmin_s32_z(svptrue_b32(), input6, inputPerm6));
            input7 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input7, inputPerm7),
                              svmin_s32_z(svptrue_b32(), input7, inputPerm7));
            input8 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input8, inputPerm8),
                              svmin_s32_z(svptrue_b32(), input8, inputPerm8));
        }
    }
#endif
}

inline void CoreSmallSort8(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                           svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8){
#ifdef NOOPTIM
    CoreSmallSort4(input, input2, input3, input4);
    CoreSmallSort4(input5, input6, input7, input8);
#else
    {
        const int nbValuesInVec = svcntw();

        const svint32_t vecindex = svindex_s32(0, 1);

        svbool_t ffttout = svzip1_b32(svpfalse_b(),svptrue_b32());

        svint32_t vecincout = svsel_s32(ffttout, svdup_s32(-1), svdup_s32(1));

        {// stepout == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));

            input2 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input2, input2Perm),
                              svmin_s32_z(svptrue_b32(), input2, input2Perm));
            input3 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input3, input3Perm),
                              svmin_s32_z(svptrue_b32(), input3, input3Perm));
            input4 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input4, input4Perm),
                              svmin_s32_z(svptrue_b32(), input4, input4Perm));

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t input8Perm = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
            input5 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input5, input5Perm),
                              svmin_s32_z(svptrue_b32(), input5, input5Perm));
            input6 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input6, input6Perm),
                              svmin_s32_z(svptrue_b32(), input6, input6Perm));
            input7 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input7, input7Perm),
                              svmin_s32_z(svptrue_b32(), input7, input7Perm));
            input8 = svsel_s32(ffttout,
                              svmax_s32_z(svptrue_b32(), input8, input8Perm),
                              svmin_s32_z(svptrue_b32(), input8, input8Perm));
        }
        for(long int stepout = 2 ; stepout < nbValuesInVec ; stepout *= 2){
            ffttout = svzip1_b32(ffttout,ffttout);
            vecincout = svsel_s32(ffttout,
                                      svsub_n_s32_z(svptrue_b32(), vecincout, stepout),
                                      svadd_n_s32_z(svptrue_b32(), vecincout, stepout));
            {
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, vecincout);

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
                input4 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input4, input4Perm),
                                  svmin_s32_z(svptrue_b32(), input4, input4Perm));

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
                const svint32_t input8Perm = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
                input5 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input5, input5Perm),
                                  svmin_s32_z(svptrue_b32(), input5, input5Perm));
                input6 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input6, input6Perm),
                                  svmin_s32_z(svptrue_b32(), input6, input6Perm));
                input7 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input7, input7Perm),
                                  svmin_s32_z(svptrue_b32(), input7, input7Perm));
                input8 = svsel_s32(ffttout,
                                  svmax_s32_z(svptrue_b32(), input8, input8Perm),
                                  svmin_s32_z(svptrue_b32(), input8, input8Perm));
            }

            svbool_t fftt = svuzp2_b32(ffttout,ffttout);

            svint32_t vecinc = svdup_s32(stepout/2);

            for(long int step = stepout/2 ; step > 1 ; step/=2){
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
                input4 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input4, input4Perm),
                                  svmin_s32_z(svptrue_b32(), input4, input4Perm));

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
                const svint32_t input8Perm = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
                input5 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input5, input5Perm),
                                  svmin_s32_z(svptrue_b32(), input5, input5Perm));
                input6 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input6, input6Perm),
                                  svmin_s32_z(svptrue_b32(), input6, input6Perm));
                input7 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input7, input7Perm),
                                  svmin_s32_z(svptrue_b32(), input7, input7Perm));
                input8 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input8, input8Perm),
                                  svmin_s32_z(svptrue_b32(), input8, input8Perm));

                fftt = svuzp2_b32(fftt,fftt);
                vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
            }
            { // Step == 1
                const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

                const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
                const svint32_t input2Perm = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
                const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
                const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

                input = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input, inputPerm),
                                  svmin_s32_z(svptrue_b32(), input, inputPerm));
                input2 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input2, input2Perm),
                                  svmin_s32_z(svptrue_b32(), input2, input2Perm));
                input3 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input3, input3Perm),
                                  svmin_s32_z(svptrue_b32(), input3, input3Perm));
                input4 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input4, input4Perm),
                                  svmin_s32_z(svptrue_b32(), input4, input4Perm));

                const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
                const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
                const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
                const svint32_t input8Perm = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
                input5 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input5, input5Perm),
                                  svmin_s32_z(svptrue_b32(), input5, input5Perm));
                input6 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input6, input6Perm),
                                  svmin_s32_z(svptrue_b32(), input6, input6Perm));
                input7 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input7, input7Perm),
                                  svmin_s32_z(svptrue_b32(), input7, input7Perm));
                input8 = svsel_s32(fftt,
                                  svmax_s32_z(svptrue_b32(), input8, input8Perm),
                                  svmin_s32_z(svptrue_b32(), input8, input8Perm));
            }
        }
    }
    {
        CoreExchange(input, input2);
    }
    {
        CoreExchange(input3, input4);
    }
    {
        CoreExchange(input5, input6);
    }
    {
        CoreExchange(input7, input8);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, input3Perm),
                              svmin_s32_z(svptrue_b32(), input3, input3Perm));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, input4Perm),
                              svmin_s32_z(svptrue_b32(), input4, input4Perm));

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t input8Perm = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, input5Perm),
                              svmin_s32_z(svptrue_b32(), input5, input5Perm));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, input6Perm),
                              svmin_s32_z(svptrue_b32(), input6, input6Perm));
            input7 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input7, input7Perm),
                              svmin_s32_z(svptrue_b32(), input7, input7Perm));
            input8 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input8, input8Perm),
                              svmin_s32_z(svptrue_b32(), input8, input8Perm));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t input3Perm = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t input4Perm = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, input3Perm),
                              svmin_s32_z(svptrue_b32(), input3, input3Perm));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, input4Perm),
                              svmin_s32_z(svptrue_b32(), input4, input4Perm));

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t input8Perm = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, input5Perm),
                              svmin_s32_z(svptrue_b32(), input5, input5Perm));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, input6Perm),
                              svmin_s32_z(svptrue_b32(), input6, input6Perm));
            input7 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input7, input7Perm),
                              svmin_s32_z(svptrue_b32(), input7, input7Perm));
            input8 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input8, input8Perm),
                              svmin_s32_z(svptrue_b32(), input8, input8Perm));
        }
    }
    {
        CoreExchange(input, input4);
        CoreExchange(input2, input3);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchange(input5, input8);
        CoreExchange(input6, input7);
    }
    {
        CoreExchangeEnd(input5, input6);
        CoreExchangeEnd(input7, input8);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t input8Perm = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, input5Perm),
                              svmin_s32_z(svptrue_b32(), input5, input5Perm));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, input6Perm),
                              svmin_s32_z(svptrue_b32(), input6, input6Perm));
            input7 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input7, input7Perm),
                              svmin_s32_z(svptrue_b32(), input7, input7Perm));
            input8 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input8, input8Perm),
                              svmin_s32_z(svptrue_b32(), input8, input8Perm));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));

            const svint32_t input5Perm = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t input6Perm = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t input7Perm = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t input8Perm = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, input5Perm),
                              svmin_s32_z(svptrue_b32(), input5, input5Perm));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, input6Perm),
                              svmin_s32_z(svptrue_b32(), input6, input6Perm));
            input7 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input7, input7Perm),
                              svmin_s32_z(svptrue_b32(), input7, input7Perm));
            input8 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input8, input8Perm),
                              svmin_s32_z(svptrue_b32(), input8, input8Perm));
        }
    }
#endif
    {
        CoreExchange(input, input8);
        CoreExchange(input2, input7);
        CoreExchange(input3, input6);
        CoreExchange(input4, input5);
    }
#ifdef NOOPTIM
    CoreSmallEnd4(input, input2, input3, input4);
    CoreSmallEnd4(input5, input6, input7, input8);
#else
    {
        CoreExchangeEnd(input, input3);
        CoreExchangeEnd(input2, input4);
    }
    {
        CoreExchangeEnd(input, input2);
        CoreExchangeEnd(input3, input4);
    }
    {
        CoreExchangeEnd(input5, input7);
        CoreExchangeEnd(input6, input8);
    }
    {
        CoreExchangeEnd(input5, input6);
        CoreExchangeEnd(input7, input8);
    }
    {
        const int nbValuesInVec = svcntw();

        svbool_t fftt = getFalseTrueMask32(nbValuesInVec/2);

        const svint32_t vecindex = svindex_s32(0, 1);

        svint32_t vecinc = svdup_s32(nbValuesInVec/2);

        for(long int step = nbValuesInVec/2 ; step > 1 ; step/=2){
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7 = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm8 = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, inputPerm5),
                              svmin_s32_z(svptrue_b32(), input5, inputPerm5));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, inputPerm6),
                              svmin_s32_z(svptrue_b32(), input6, inputPerm6));
            input7 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input7, inputPerm7),
                              svmin_s32_z(svptrue_b32(), input7, inputPerm7));
            input8 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input8, inputPerm8),
                              svmin_s32_z(svptrue_b32(), input8, inputPerm8));

            fftt = svuzp2_b32(fftt,fftt);
            vecinc = svdiv_n_s32_z(svptrue_b32(), vecinc, 2); // Could be shift be need cast
        }
        { // Step == 1
            const svint32_t vecpermute = svadd_s32_z(svptrue_b32(), vecindex, svsel_s32(fftt, svneg_s32_z(fftt, vecinc), vecinc));

            const svint32_t inputPerm = svtbl_s32(input, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm2 = svtbl_s32(input2, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm3 = svtbl_s32(input3, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm4 = svtbl_s32(input4, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm5 = svtbl_s32(input5, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm6 = svtbl_s32(input6, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm7 = svtbl_s32(input7, svreinterpret_u32_s32(vecpermute));
            const svint32_t inputPerm8 = svtbl_s32(input8, svreinterpret_u32_s32(vecpermute));

            input = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input, inputPerm),
                              svmin_s32_z(svptrue_b32(), input, inputPerm));
            input2 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input2, inputPerm2),
                              svmin_s32_z(svptrue_b32(), input2, inputPerm2));
            input3 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input3, inputPerm3),
                              svmin_s32_z(svptrue_b32(), input3, inputPerm3));
            input4 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input4, inputPerm4),
                              svmin_s32_z(svptrue_b32(), input4, inputPerm4));
            input5 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input5, inputPerm5),
                              svmin_s32_z(svptrue_b32(), input5, inputPerm5));
            input6 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input6, inputPerm6),
                              svmin_s32_z(svptrue_b32(), input6, inputPerm6));
            input7 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input7, inputPerm7),
                              svmin_s32_z(svptrue_b32(), input7, inputPerm7));
            input8 = svsel_s32(fftt,
                              svmax_s32_z(svptrue_b32(), input8, inputPerm8),
                              svmin_s32_z(svptrue_b32(), input8, inputPerm8));
        }
    }
#endif
}

inline void CoreSmallSort9(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                           svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                           svint32_t& input9){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort(input9);
    {
        CoreExchange(input8, input9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSortEnd(input9);
}

inline void CoreSmallSort10(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort2(input9, input10);
    {
        CoreExchange(input7, input10);
        CoreExchange(input8, input9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd2(input9, input10);
}

inline void CoreSmallSort11(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort3(input9, input10, input11);
    {
        CoreExchange(input6, input11);
        CoreExchange(input7, input10);
        CoreExchange(input8, input9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd3(input9, input10, input11);
}

inline void CoreSmallSort12(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort4(input9, input10, input11, input12);
    {
        CoreExchange(input5, input12);
        CoreExchange(input6, input11);
        CoreExchange(input7, input10);
        CoreExchange(input8, input9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd4(input9, input10, input11, input12);
}

inline void CoreSmallSort13(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                            svint32_t& input13){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort5(input9, input10, input11, input12, input13);
    {
        CoreExchange(input4, input13);
        CoreExchange(input5, input12);
        CoreExchange(input6, input11);
        CoreExchange(input7, input10);
        CoreExchange(input8, input9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd5(input9, input10, input11, input12, input13);
}

inline void CoreSmallSort14(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                            svint32_t& input13, svint32_t& input14){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort6(input9, input10, input11, input12, input13, input14);
    {
        CoreExchange(input3, input14);
        CoreExchange(input4, input13);
        CoreExchange(input5, input12);
        CoreExchange(input6, input11);
        CoreExchange(input7, input10);
        CoreExchange(input8, input9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd6(input9, input10, input11, input12, input13, input14);
}

inline void CoreSmallSort15(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                            svint32_t& input13, svint32_t& input14, svint32_t& input15){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort7(input9, input10, input11, input12, input13, input14, input15);
    {
        CoreExchange(input2, input15);
        CoreExchange(input3, input14);
        CoreExchange(input4, input13);
        CoreExchange(input5, input12);
        CoreExchange(input6, input11);
        CoreExchange(input7, input10);
        CoreExchange(input8, input9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd7(input9, input10, input11, input12, input13, input14, input15);
}

inline void CoreSmallSort16(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                            svint32_t& input13, svint32_t& input14, svint32_t& input15, svint32_t& input16){
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallSort8(input9, input10, input11, input12, input13, input14, input15, input16);
    {
        CoreExchange(input, input16);
        CoreExchange(input2, input15);
        CoreExchange(input3, input14);
        CoreExchange(input4, input13);
        CoreExchange(input5, input12);
        CoreExchange(input6, input11);
        CoreExchange(input7, input10);
        CoreExchange(input8, input9);
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8);
    CoreSmallEnd8(input9, input10, input11, input12, input13, input14, input15, input16);
}

////////////////////////////////////////////////////////////

inline void CoreSmallSort1( int* __restrict__ ptr1){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    CoreSmallSort(v1);
    svst1_s32(svptrue_b32(), ptr1, v1);
}

inline bool IsSorted( int* __restrict__ ptr1){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    return IsSorted(v1);
}


inline void CoreSmallSort1( double* __restrict__ ptr1){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    CoreSmallSort(v1);
    svst1_f64(svptrue_b64(), ptr1, v1);
}

inline bool IsSorted( double* __restrict__ ptr1){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    return IsSorted(v1);
}


inline void CoreSmallSort2( int* __restrict__ ptr1,int* __restrict__ ptr2){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t v2 = svld1_s32(svptrue_b32(),ptr2);
    CoreSmallSort2(v1,v2);
    svst1_s32(svptrue_b32(), ptr1, v1);
    svst1_s32(svptrue_b32(), ptr2, v2);
}


inline void CoreSmallSort2( double* __restrict__ ptr1,double* __restrict__ ptr2){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr2);
    CoreSmallSort2(v1,v2);
    svst1_f64(svptrue_b64(), ptr1, v1);
    svst1_f64(svptrue_b64(), ptr2, v2);
}


inline void CoreSmallSort3( int* __restrict__ ptr1,int* __restrict__ ptr2,int* __restrict__ ptr3){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t v2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t v3 = svld1_s32(svptrue_b32(),ptr3);
    CoreSmallSort3(v1,v2,v3);
    svst1_s32(svptrue_b32(), ptr1, v1);
    svst1_s32(svptrue_b32(), ptr2, v2);
    svst1_s32(svptrue_b32(), ptr3, v3);
}


inline void CoreSmallSort3( double* __restrict__ ptr1,double* __restrict__ ptr2,double* __restrict__ ptr3){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr3);
    CoreSmallSort3(v1,v2,v3);
    svst1_f64(svptrue_b64(), ptr1, v1);
    svst1_f64(svptrue_b64(), ptr2, v2);
    svst1_f64(svptrue_b64(), ptr3, v3);
}


inline void CoreSmallSort4( int* __restrict__ ptr1,int* __restrict__ ptr2,int* __restrict__ ptr3,
                            int* __restrict__ ptr4){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t v2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t v3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t v4 = svld1_s32(svptrue_b32(),ptr4);
    CoreSmallSort4(v1,v2,v3,
                   v4);
    svst1_s32(svptrue_b32(), ptr1, v1);
    svst1_s32(svptrue_b32(), ptr2, v2);
    svst1_s32(svptrue_b32(), ptr3, v3);
    svst1_s32(svptrue_b32(), ptr4, v4);
}


inline void CoreSmallSort4( double* __restrict__ ptr1,double* __restrict__ ptr2,double* __restrict__ ptr3,
                            double* __restrict__ ptr4){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr4);
    CoreSmallSort4(v1,v2,v3,
                   v4);
    svst1_f64(svptrue_b64(), ptr1, v1);
    svst1_f64(svptrue_b64(), ptr2, v2);
    svst1_f64(svptrue_b64(), ptr3, v3);
    svst1_f64(svptrue_b64(), ptr4, v4);
}


inline void CoreSmallSort5( int* __restrict__ ptr1,int* __restrict__ ptr2,int* __restrict__ ptr3,
                            int* __restrict__ ptr4,int* __restrict__ ptr5){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t v2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t v3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t v4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t v5 = svld1_s32(svptrue_b32(),ptr5);
    CoreSmallSort5(v1,v2,v3,
                   v4,v5);
    svst1_s32(svptrue_b32(), ptr1, v1);
    svst1_s32(svptrue_b32(), ptr2, v2);
    svst1_s32(svptrue_b32(), ptr3, v3);
    svst1_s32(svptrue_b32(), ptr4, v4);
    svst1_s32(svptrue_b32(), ptr5, v5);
}


inline void CoreSmallSort5( double* __restrict__ ptr1,double* __restrict__ ptr2,double* __restrict__ ptr3,
                            double* __restrict__ ptr4,double* __restrict__ ptr5){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr5);
    CoreSmallSort5(v1,v2,v3,
                   v4,v5);
    svst1_f64(svptrue_b64(), ptr1, v1);
    svst1_f64(svptrue_b64(), ptr2, v2);
    svst1_f64(svptrue_b64(), ptr3, v3);
    svst1_f64(svptrue_b64(), ptr4, v4);
    svst1_f64(svptrue_b64(), ptr5, v5);
}


inline void CoreSmallSort6( int* __restrict__ ptr1,int* __restrict__ ptr2,int* __restrict__ ptr3,
                            int* __restrict__ ptr4,int* __restrict__ ptr5,int* __restrict__ ptr6){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t v2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t v3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t v4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t v5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t v6 = svld1_s32(svptrue_b32(),ptr6);
    CoreSmallSort6(v1,v2,v3,
                   v4,v5,v6);
    svst1_s32(svptrue_b32(), ptr1, v1);
    svst1_s32(svptrue_b32(), ptr2, v2);
    svst1_s32(svptrue_b32(), ptr3, v3);
    svst1_s32(svptrue_b32(), ptr4, v4);
    svst1_s32(svptrue_b32(), ptr5, v5);
    svst1_s32(svptrue_b32(), ptr6, v6);
}


inline void CoreSmallSort6( double* __restrict__ ptr1,double* __restrict__ ptr2,double* __restrict__ ptr3,
                            double* __restrict__ ptr4,double* __restrict__ ptr5,double* __restrict__ ptr6){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr6);
    CoreSmallSort6(v1,v2,v3,
                   v4,v5,v6);
    svst1_f64(svptrue_b64(), ptr1, v1);
    svst1_f64(svptrue_b64(), ptr2, v2);
    svst1_f64(svptrue_b64(), ptr3, v3);
    svst1_f64(svptrue_b64(), ptr4, v4);
    svst1_f64(svptrue_b64(), ptr5, v5);
    svst1_f64(svptrue_b64(), ptr6, v6);
}


inline void CoreSmallSort7( int* __restrict__ ptr1,int* __restrict__ ptr2,int* __restrict__ ptr3,
                            int* __restrict__ ptr4,int* __restrict__ ptr5,int* __restrict__ ptr6,int* __restrict__ ptr7){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t v2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t v3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t v4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t v5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t v6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t v7 = svld1_s32(svptrue_b32(),ptr7);
    CoreSmallSort7(v1,v2,v3,
                   v4,v5,v6,v7);
    svst1_s32(svptrue_b32(), ptr1, v1);
    svst1_s32(svptrue_b32(), ptr2, v2);
    svst1_s32(svptrue_b32(), ptr3, v3);
    svst1_s32(svptrue_b32(), ptr4, v4);
    svst1_s32(svptrue_b32(), ptr5, v5);
    svst1_s32(svptrue_b32(), ptr6, v6);
    svst1_s32(svptrue_b32(), ptr7, v7);
}


inline void CoreSmallSort7( double* __restrict__ ptr1,double* __restrict__ ptr2,double* __restrict__ ptr3,
                            double* __restrict__ ptr4,double* __restrict__ ptr5,double* __restrict__ ptr6,double* __restrict__ ptr7){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr7);
    CoreSmallSort7(v1,v2,v3,
                   v4,v5,v6,v7);
    svst1_f64(svptrue_b64(), ptr1, v1);
    svst1_f64(svptrue_b64(), ptr2, v2);
    svst1_f64(svptrue_b64(), ptr3, v3);
    svst1_f64(svptrue_b64(), ptr4, v4);
    svst1_f64(svptrue_b64(), ptr5, v5);
    svst1_f64(svptrue_b64(), ptr6, v6);
    svst1_f64(svptrue_b64(), ptr7, v7);
}


inline void CoreSmallSort8( int* __restrict__ ptr1,int* __restrict__ ptr2,int* __restrict__ ptr3,
                            int* __restrict__ ptr4,int* __restrict__ ptr5,int* __restrict__ ptr6,int* __restrict__ ptr7,
                            int* __restrict__ ptr8){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t v2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t v3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t v4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t v5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t v6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t v7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t v8 = svld1_s32(svptrue_b32(),ptr8);
    CoreSmallSort8(v1,v2,v3,
                   v4,v5,v6,v7,
                   v8);
    svst1_s32(svptrue_b32(), ptr1, v1);
    svst1_s32(svptrue_b32(), ptr2, v2);
    svst1_s32(svptrue_b32(), ptr3, v3);
    svst1_s32(svptrue_b32(), ptr4, v4);
    svst1_s32(svptrue_b32(), ptr5, v5);
    svst1_s32(svptrue_b32(), ptr6, v6);
    svst1_s32(svptrue_b32(), ptr7, v7);
    svst1_s32(svptrue_b32(), ptr8, v8);
}


inline void CoreSmallSort8( double* __restrict__ ptr1,double* __restrict__ ptr2,double* __restrict__ ptr3,
                            double* __restrict__ ptr4,double* __restrict__ ptr5,double* __restrict__ ptr6,double* __restrict__ ptr7,
                            double* __restrict__ ptr8){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr8);
    CoreSmallSort8(v1,v2,v3,
                   v4,v5,v6,v7,
                   v8);
    svst1_f64(svptrue_b64(), ptr1, v1);
    svst1_f64(svptrue_b64(), ptr2, v2);
    svst1_f64(svptrue_b64(), ptr3, v3);
    svst1_f64(svptrue_b64(), ptr4, v4);
    svst1_f64(svptrue_b64(), ptr5, v5);
    svst1_f64(svptrue_b64(), ptr6, v6);
    svst1_f64(svptrue_b64(), ptr7, v7);
    svst1_f64(svptrue_b64(), ptr8, v8);
}


inline void CoreSmallSort9( int* __restrict__ ptr1,int* __restrict__ ptr2,int* __restrict__ ptr3,
                            int* __restrict__ ptr4,int* __restrict__ ptr5,int* __restrict__ ptr6,int* __restrict__ ptr7,
                            int* __restrict__ ptr8,int* __restrict__ ptr9){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t v2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t v3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t v4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t v5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t v6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t v7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t v8 = svld1_s32(svptrue_b32(),ptr8);
    svint32_t v9 = svld1_s32(svptrue_b32(),ptr9);
    CoreSmallSort9(v1,v2,v3,
                   v4,v5,v6,v7,
                   v8,v9);
    svst1_s32(svptrue_b32(), ptr1, v1);
    svst1_s32(svptrue_b32(), ptr2, v2);
    svst1_s32(svptrue_b32(), ptr3, v3);
    svst1_s32(svptrue_b32(), ptr4, v4);
    svst1_s32(svptrue_b32(), ptr5, v5);
    svst1_s32(svptrue_b32(), ptr6, v6);
    svst1_s32(svptrue_b32(), ptr7, v7);
    svst1_s32(svptrue_b32(), ptr8, v8);
    svst1_s32(svptrue_b32(), ptr9, v9);
}


inline void CoreSmallSort9( double* __restrict__ ptr1,double* __restrict__ ptr2,double* __restrict__ ptr3,
                            double* __restrict__ ptr4,double* __restrict__ ptr5,double* __restrict__ ptr6,double* __restrict__ ptr7,
                            double* __restrict__ ptr8,double* __restrict__ ptr9){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr8);
    svfloat64_t v9 = svld1_f64(svptrue_b64(),ptr9);
    CoreSmallSort9(v1,v2,v3,
                   v4,v5,v6,v7,
                   v8,v9);
    svst1_f64(svptrue_b64(), ptr1, v1);
    svst1_f64(svptrue_b64(), ptr2, v2);
    svst1_f64(svptrue_b64(), ptr3, v3);
    svst1_f64(svptrue_b64(), ptr4, v4);
    svst1_f64(svptrue_b64(), ptr5, v5);
    svst1_f64(svptrue_b64(), ptr6, v6);
    svst1_f64(svptrue_b64(), ptr7, v7);
    svst1_f64(svptrue_b64(), ptr8, v8);
    svst1_f64(svptrue_b64(), ptr9, v9);
}


inline void CoreSmallSort10( int* __restrict__ ptr1,int* __restrict__ ptr2,int* __restrict__ ptr3,
                             int* __restrict__ ptr4,int* __restrict__ ptr5,int* __restrict__ ptr6,int* __restrict__ ptr7,
                             int* __restrict__ ptr8,int* __restrict__ ptr9,int* __restrict__ ptr10){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t v2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t v3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t v4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t v5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t v6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t v7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t v8 = svld1_s32(svptrue_b32(),ptr8);
    svint32_t v9 = svld1_s32(svptrue_b32(),ptr9);
    svint32_t v10 = svld1_s32(svptrue_b32(),ptr10);
    CoreSmallSort10(v1,v2,v3,
                    v4,v5,v6,v7,
                    v8,v9,v10);
    svst1_s32(svptrue_b32(), ptr1, v1);
    svst1_s32(svptrue_b32(), ptr2, v2);
    svst1_s32(svptrue_b32(), ptr3, v3);
    svst1_s32(svptrue_b32(), ptr4, v4);
    svst1_s32(svptrue_b32(), ptr5, v5);
    svst1_s32(svptrue_b32(), ptr6, v6);
    svst1_s32(svptrue_b32(), ptr7, v7);
    svst1_s32(svptrue_b32(), ptr8, v8);
    svst1_s32(svptrue_b32(), ptr9, v9);
    svst1_s32(svptrue_b32(), ptr10, v10);
}


inline void CoreSmallSort10( double* __restrict__ ptr1,double* __restrict__ ptr2,double* __restrict__ ptr3,
                             double* __restrict__ ptr4,double* __restrict__ ptr5,double* __restrict__ ptr6,double* __restrict__ ptr7,
                             double* __restrict__ ptr8,double* __restrict__ ptr9,double* __restrict__ ptr10){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr8);
    svfloat64_t v9 = svld1_f64(svptrue_b64(),ptr9);
    svfloat64_t v10 = svld1_f64(svptrue_b64(),ptr10);
    CoreSmallSort10(v1,v2,v3,
                    v4,v5,v6,v7,
                    v8,v9,v10);
    svst1_f64(svptrue_b64(), ptr1, v1);
    svst1_f64(svptrue_b64(), ptr2, v2);
    svst1_f64(svptrue_b64(), ptr3, v3);
    svst1_f64(svptrue_b64(), ptr4, v4);
    svst1_f64(svptrue_b64(), ptr5, v5);
    svst1_f64(svptrue_b64(), ptr6, v6);
    svst1_f64(svptrue_b64(), ptr7, v7);
    svst1_f64(svptrue_b64(), ptr8, v8);
    svst1_f64(svptrue_b64(), ptr9, v9);
    svst1_f64(svptrue_b64(), ptr10, v10);
}


inline void CoreSmallSort11( int* __restrict__ ptr1,int* __restrict__ ptr2,int* __restrict__ ptr3,
                             int* __restrict__ ptr4,int* __restrict__ ptr5,int* __restrict__ ptr6,int* __restrict__ ptr7,
                             int* __restrict__ ptr8,int* __restrict__ ptr9,int* __restrict__ ptr10,int* __restrict__ ptr11){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t v2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t v3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t v4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t v5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t v6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t v7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t v8 = svld1_s32(svptrue_b32(),ptr8);
    svint32_t v9 = svld1_s32(svptrue_b32(),ptr9);
    svint32_t v10 = svld1_s32(svptrue_b32(),ptr10);
    svint32_t v11 = svld1_s32(svptrue_b32(),ptr11);
    CoreSmallSort11(v1,v2,v3,
                    v4,v5,v6,v7,
                    v8,v9,v10,v11);
    svst1_s32(svptrue_b32(), ptr1, v1);
    svst1_s32(svptrue_b32(), ptr2, v2);
    svst1_s32(svptrue_b32(), ptr3, v3);
    svst1_s32(svptrue_b32(), ptr4, v4);
    svst1_s32(svptrue_b32(), ptr5, v5);
    svst1_s32(svptrue_b32(), ptr6, v6);
    svst1_s32(svptrue_b32(), ptr7, v7);
    svst1_s32(svptrue_b32(), ptr8, v8);
    svst1_s32(svptrue_b32(), ptr9, v9);
    svst1_s32(svptrue_b32(), ptr10, v10);
    svst1_s32(svptrue_b32(), ptr11, v11);
}


inline void CoreSmallSort11( double* __restrict__ ptr1,double* __restrict__ ptr2,double* __restrict__ ptr3,
                             double* __restrict__ ptr4,double* __restrict__ ptr5,double* __restrict__ ptr6,double* __restrict__ ptr7,
                             double* __restrict__ ptr8,double* __restrict__ ptr9,double* __restrict__ ptr10,double* __restrict__ ptr11){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr8);
    svfloat64_t v9 = svld1_f64(svptrue_b64(),ptr9);
    svfloat64_t v10 = svld1_f64(svptrue_b64(),ptr10);
    svfloat64_t v11 = svld1_f64(svptrue_b64(),ptr11);
    CoreSmallSort11(v1,v2,v3,
                    v4,v5,v6,v7,
                    v8,v9,v10,v11);
    svst1_f64(svptrue_b64(), ptr1, v1);
    svst1_f64(svptrue_b64(), ptr2, v2);
    svst1_f64(svptrue_b64(), ptr3, v3);
    svst1_f64(svptrue_b64(), ptr4, v4);
    svst1_f64(svptrue_b64(), ptr5, v5);
    svst1_f64(svptrue_b64(), ptr6, v6);
    svst1_f64(svptrue_b64(), ptr7, v7);
    svst1_f64(svptrue_b64(), ptr8, v8);
    svst1_f64(svptrue_b64(), ptr9, v9);
    svst1_f64(svptrue_b64(), ptr10, v10);
    svst1_f64(svptrue_b64(), ptr11, v11);
}


inline void CoreSmallSort12( int* __restrict__ ptr1,int* __restrict__ ptr2,int* __restrict__ ptr3,
                             int* __restrict__ ptr4,int* __restrict__ ptr5,int* __restrict__ ptr6,int* __restrict__ ptr7,
                             int* __restrict__ ptr8,int* __restrict__ ptr9,int* __restrict__ ptr10,int* __restrict__ ptr11,
                             int* __restrict__ ptr12){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t v2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t v3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t v4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t v5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t v6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t v7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t v8 = svld1_s32(svptrue_b32(),ptr8);
    svint32_t v9 = svld1_s32(svptrue_b32(),ptr9);
    svint32_t v10 = svld1_s32(svptrue_b32(),ptr10);
    svint32_t v11 = svld1_s32(svptrue_b32(),ptr11);
    svint32_t v12 = svld1_s32(svptrue_b32(),ptr12);
    CoreSmallSort12(v1,v2,v3,
                    v4,v5,v6,v7,
                    v8,v9,v10,v11,
                    v12);
    svst1_s32(svptrue_b32(), ptr1, v1);
    svst1_s32(svptrue_b32(), ptr2, v2);
    svst1_s32(svptrue_b32(), ptr3, v3);
    svst1_s32(svptrue_b32(), ptr4, v4);
    svst1_s32(svptrue_b32(), ptr5, v5);
    svst1_s32(svptrue_b32(), ptr6, v6);
    svst1_s32(svptrue_b32(), ptr7, v7);
    svst1_s32(svptrue_b32(), ptr8, v8);
    svst1_s32(svptrue_b32(), ptr9, v9);
    svst1_s32(svptrue_b32(), ptr10, v10);
    svst1_s32(svptrue_b32(), ptr11, v11);
    svst1_s32(svptrue_b32(), ptr12, v12);
}


inline void CoreSmallSort12( double* __restrict__ ptr1,double* __restrict__ ptr2,double* __restrict__ ptr3,
                             double* __restrict__ ptr4,double* __restrict__ ptr5,double* __restrict__ ptr6,double* __restrict__ ptr7,
                             double* __restrict__ ptr8,double* __restrict__ ptr9,double* __restrict__ ptr10,double* __restrict__ ptr11,
                             double* __restrict__ ptr12){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr8);
    svfloat64_t v9 = svld1_f64(svptrue_b64(),ptr9);
    svfloat64_t v10 = svld1_f64(svptrue_b64(),ptr10);
    svfloat64_t v11 = svld1_f64(svptrue_b64(),ptr11);
    svfloat64_t v12 = svld1_f64(svptrue_b64(),ptr12);
    CoreSmallSort12(v1,v2,v3,
                    v4,v5,v6,v7,
                    v8,v9,v10,v11,
                    v12);
    svst1_f64(svptrue_b64(), ptr1, v1);
    svst1_f64(svptrue_b64(), ptr2, v2);
    svst1_f64(svptrue_b64(), ptr3, v3);
    svst1_f64(svptrue_b64(), ptr4, v4);
    svst1_f64(svptrue_b64(), ptr5, v5);
    svst1_f64(svptrue_b64(), ptr6, v6);
    svst1_f64(svptrue_b64(), ptr7, v7);
    svst1_f64(svptrue_b64(), ptr8, v8);
    svst1_f64(svptrue_b64(), ptr9, v9);
    svst1_f64(svptrue_b64(), ptr10, v10);
    svst1_f64(svptrue_b64(), ptr11, v11);
    svst1_f64(svptrue_b64(), ptr12, v12);
}


inline void CoreSmallSort13( int* __restrict__ ptr1,int* __restrict__ ptr2,int* __restrict__ ptr3,
                             int* __restrict__ ptr4,int* __restrict__ ptr5,int* __restrict__ ptr6,int* __restrict__ ptr7,
                             int* __restrict__ ptr8,int* __restrict__ ptr9,int* __restrict__ ptr10,int* __restrict__ ptr11,
                             int* __restrict__ ptr12,int* __restrict__ ptr13){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t v2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t v3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t v4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t v5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t v6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t v7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t v8 = svld1_s32(svptrue_b32(),ptr8);
    svint32_t v9 = svld1_s32(svptrue_b32(),ptr9);
    svint32_t v10 = svld1_s32(svptrue_b32(),ptr10);
    svint32_t v11 = svld1_s32(svptrue_b32(),ptr11);
    svint32_t v12 = svld1_s32(svptrue_b32(),ptr12);
    svint32_t v13 = svld1_s32(svptrue_b32(),ptr13);
    CoreSmallSort13(v1,v2,v3,
                    v4,v5,v6,v7,
                    v8,v9,v10,v11,
                    v12,v13);
    svst1_s32(svptrue_b32(), ptr1, v1);
    svst1_s32(svptrue_b32(), ptr2, v2);
    svst1_s32(svptrue_b32(), ptr3, v3);
    svst1_s32(svptrue_b32(), ptr4, v4);
    svst1_s32(svptrue_b32(), ptr5, v5);
    svst1_s32(svptrue_b32(), ptr6, v6);
    svst1_s32(svptrue_b32(), ptr7, v7);
    svst1_s32(svptrue_b32(), ptr8, v8);
    svst1_s32(svptrue_b32(), ptr9, v9);
    svst1_s32(svptrue_b32(), ptr10, v10);
    svst1_s32(svptrue_b32(), ptr11, v11);
    svst1_s32(svptrue_b32(), ptr12, v12);
    svst1_s32(svptrue_b32(), ptr13, v13);
}


inline void CoreSmallSort13( double* __restrict__ ptr1,double* __restrict__ ptr2,double* __restrict__ ptr3,
                             double* __restrict__ ptr4,double* __restrict__ ptr5,double* __restrict__ ptr6,double* __restrict__ ptr7,
                             double* __restrict__ ptr8,double* __restrict__ ptr9,double* __restrict__ ptr10,double* __restrict__ ptr11,
                             double* __restrict__ ptr12,double* __restrict__ ptr13){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr8);
    svfloat64_t v9 = svld1_f64(svptrue_b64(),ptr9);
    svfloat64_t v10 = svld1_f64(svptrue_b64(),ptr10);
    svfloat64_t v11 = svld1_f64(svptrue_b64(),ptr11);
    svfloat64_t v12 = svld1_f64(svptrue_b64(),ptr12);
    svfloat64_t v13 = svld1_f64(svptrue_b64(),ptr13);
    CoreSmallSort13(v1,v2,v3,
                    v4,v5,v6,v7,
                    v8,v9,v10,v11,
                    v12,v13);
    svst1_f64(svptrue_b64(), ptr1, v1);
    svst1_f64(svptrue_b64(), ptr2, v2);
    svst1_f64(svptrue_b64(), ptr3, v3);
    svst1_f64(svptrue_b64(), ptr4, v4);
    svst1_f64(svptrue_b64(), ptr5, v5);
    svst1_f64(svptrue_b64(), ptr6, v6);
    svst1_f64(svptrue_b64(), ptr7, v7);
    svst1_f64(svptrue_b64(), ptr8, v8);
    svst1_f64(svptrue_b64(), ptr9, v9);
    svst1_f64(svptrue_b64(), ptr10, v10);
    svst1_f64(svptrue_b64(), ptr11, v11);
    svst1_f64(svptrue_b64(), ptr12, v12);
    svst1_f64(svptrue_b64(), ptr13, v13);
}


inline void CoreSmallSort14( int* __restrict__ ptr1,int* __restrict__ ptr2,int* __restrict__ ptr3,
                             int* __restrict__ ptr4,int* __restrict__ ptr5,int* __restrict__ ptr6,int* __restrict__ ptr7,
                             int* __restrict__ ptr8,int* __restrict__ ptr9,int* __restrict__ ptr10,int* __restrict__ ptr11,
                             int* __restrict__ ptr12,int* __restrict__ ptr13,int* __restrict__ ptr14){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t v2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t v3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t v4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t v5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t v6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t v7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t v8 = svld1_s32(svptrue_b32(),ptr8);
    svint32_t v9 = svld1_s32(svptrue_b32(),ptr9);
    svint32_t v10 = svld1_s32(svptrue_b32(),ptr10);
    svint32_t v11 = svld1_s32(svptrue_b32(),ptr11);
    svint32_t v12 = svld1_s32(svptrue_b32(),ptr12);
    svint32_t v13 = svld1_s32(svptrue_b32(),ptr13);
    svint32_t v14 = svld1_s32(svptrue_b32(),ptr14);
    CoreSmallSort14(v1,v2,v3,
                    v4,v5,v6,v7,
                    v8,v9,v10,v11,
                    v12,v13,v14);
    svst1_s32(svptrue_b32(), ptr1, v1);
    svst1_s32(svptrue_b32(), ptr2, v2);
    svst1_s32(svptrue_b32(), ptr3, v3);
    svst1_s32(svptrue_b32(), ptr4, v4);
    svst1_s32(svptrue_b32(), ptr5, v5);
    svst1_s32(svptrue_b32(), ptr6, v6);
    svst1_s32(svptrue_b32(), ptr7, v7);
    svst1_s32(svptrue_b32(), ptr8, v8);
    svst1_s32(svptrue_b32(), ptr9, v9);
    svst1_s32(svptrue_b32(), ptr10, v10);
    svst1_s32(svptrue_b32(), ptr11, v11);
    svst1_s32(svptrue_b32(), ptr12, v12);
    svst1_s32(svptrue_b32(), ptr13, v13);
    svst1_s32(svptrue_b32(), ptr14, v14);
}


inline void CoreSmallSort14( double* __restrict__ ptr1,double* __restrict__ ptr2,double* __restrict__ ptr3,
                             double* __restrict__ ptr4,double* __restrict__ ptr5,double* __restrict__ ptr6,double* __restrict__ ptr7,
                             double* __restrict__ ptr8,double* __restrict__ ptr9,double* __restrict__ ptr10,double* __restrict__ ptr11,
                             double* __restrict__ ptr12,double* __restrict__ ptr13,double* __restrict__ ptr14){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr8);
    svfloat64_t v9 = svld1_f64(svptrue_b64(),ptr9);
    svfloat64_t v10 = svld1_f64(svptrue_b64(),ptr10);
    svfloat64_t v11 = svld1_f64(svptrue_b64(),ptr11);
    svfloat64_t v12 = svld1_f64(svptrue_b64(),ptr12);
    svfloat64_t v13 = svld1_f64(svptrue_b64(),ptr13);
    svfloat64_t v14 = svld1_f64(svptrue_b64(),ptr14);
    CoreSmallSort14(v1,v2,v3,
                    v4,v5,v6,v7,
                    v8,v9,v10,v11,
                    v12,v13,v14);
    svst1_f64(svptrue_b64(), ptr1, v1);
    svst1_f64(svptrue_b64(), ptr2, v2);
    svst1_f64(svptrue_b64(), ptr3, v3);
    svst1_f64(svptrue_b64(), ptr4, v4);
    svst1_f64(svptrue_b64(), ptr5, v5);
    svst1_f64(svptrue_b64(), ptr6, v6);
    svst1_f64(svptrue_b64(), ptr7, v7);
    svst1_f64(svptrue_b64(), ptr8, v8);
    svst1_f64(svptrue_b64(), ptr9, v9);
    svst1_f64(svptrue_b64(), ptr10, v10);
    svst1_f64(svptrue_b64(), ptr11, v11);
    svst1_f64(svptrue_b64(), ptr12, v12);
    svst1_f64(svptrue_b64(), ptr13, v13);
    svst1_f64(svptrue_b64(), ptr14, v14);
}


inline void CoreSmallSort15( int* __restrict__ ptr1,int* __restrict__ ptr2,int* __restrict__ ptr3,
                             int* __restrict__ ptr4,int* __restrict__ ptr5,int* __restrict__ ptr6,int* __restrict__ ptr7,
                             int* __restrict__ ptr8,int* __restrict__ ptr9,int* __restrict__ ptr10,int* __restrict__ ptr11,
                             int* __restrict__ ptr12,int* __restrict__ ptr13,int* __restrict__ ptr14,int* __restrict__ ptr15){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t v2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t v3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t v4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t v5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t v6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t v7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t v8 = svld1_s32(svptrue_b32(),ptr8);
    svint32_t v9 = svld1_s32(svptrue_b32(),ptr9);
    svint32_t v10 = svld1_s32(svptrue_b32(),ptr10);
    svint32_t v11 = svld1_s32(svptrue_b32(),ptr11);
    svint32_t v12 = svld1_s32(svptrue_b32(),ptr12);
    svint32_t v13 = svld1_s32(svptrue_b32(),ptr13);
    svint32_t v14 = svld1_s32(svptrue_b32(),ptr14);
    svint32_t v15 = svld1_s32(svptrue_b32(),ptr15);
    CoreSmallSort15(v1,v2,v3,
                    v4,v5,v6,v7,
                    v8,v9,v10,v11,
                    v12,v13,v14,v15);
    svst1_s32(svptrue_b32(), ptr1, v1);
    svst1_s32(svptrue_b32(), ptr2, v2);
    svst1_s32(svptrue_b32(), ptr3, v3);
    svst1_s32(svptrue_b32(), ptr4, v4);
    svst1_s32(svptrue_b32(), ptr5, v5);
    svst1_s32(svptrue_b32(), ptr6, v6);
    svst1_s32(svptrue_b32(), ptr7, v7);
    svst1_s32(svptrue_b32(), ptr8, v8);
    svst1_s32(svptrue_b32(), ptr9, v9);
    svst1_s32(svptrue_b32(), ptr10, v10);
    svst1_s32(svptrue_b32(), ptr11, v11);
    svst1_s32(svptrue_b32(), ptr12, v12);
    svst1_s32(svptrue_b32(), ptr13, v13);
    svst1_s32(svptrue_b32(), ptr14, v14);
    svst1_s32(svptrue_b32(), ptr15, v15);
}


inline void CoreSmallSort15( double* __restrict__ ptr1,double* __restrict__ ptr2,double* __restrict__ ptr3,
                             double* __restrict__ ptr4,double* __restrict__ ptr5,double* __restrict__ ptr6,double* __restrict__ ptr7,
                             double* __restrict__ ptr8,double* __restrict__ ptr9,double* __restrict__ ptr10,double* __restrict__ ptr11,
                             double* __restrict__ ptr12,double* __restrict__ ptr13,double* __restrict__ ptr14,double* __restrict__ ptr15){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr8);
    svfloat64_t v9 = svld1_f64(svptrue_b64(),ptr9);
    svfloat64_t v10 = svld1_f64(svptrue_b64(),ptr10);
    svfloat64_t v11 = svld1_f64(svptrue_b64(),ptr11);
    svfloat64_t v12 = svld1_f64(svptrue_b64(),ptr12);
    svfloat64_t v13 = svld1_f64(svptrue_b64(),ptr13);
    svfloat64_t v14 = svld1_f64(svptrue_b64(),ptr14);
    svfloat64_t v15 = svld1_f64(svptrue_b64(),ptr15);
    CoreSmallSort15(v1,v2,v3,
                    v4,v5,v6,v7,
                    v8,v9,v10,v11,
                    v12,v13,v14,v15);
    svst1_f64(svptrue_b64(), ptr1, v1);
    svst1_f64(svptrue_b64(), ptr2, v2);
    svst1_f64(svptrue_b64(), ptr3, v3);
    svst1_f64(svptrue_b64(), ptr4, v4);
    svst1_f64(svptrue_b64(), ptr5, v5);
    svst1_f64(svptrue_b64(), ptr6, v6);
    svst1_f64(svptrue_b64(), ptr7, v7);
    svst1_f64(svptrue_b64(), ptr8, v8);
    svst1_f64(svptrue_b64(), ptr9, v9);
    svst1_f64(svptrue_b64(), ptr10, v10);
    svst1_f64(svptrue_b64(), ptr11, v11);
    svst1_f64(svptrue_b64(), ptr12, v12);
    svst1_f64(svptrue_b64(), ptr13, v13);
    svst1_f64(svptrue_b64(), ptr14, v14);
    svst1_f64(svptrue_b64(), ptr15, v15);
}


inline void CoreSmallSort16( int* __restrict__ ptr1,int* __restrict__ ptr2,int* __restrict__ ptr3,
                             int* __restrict__ ptr4,int* __restrict__ ptr5,int* __restrict__ ptr6,int* __restrict__ ptr7,
                             int* __restrict__ ptr8,int* __restrict__ ptr9,int* __restrict__ ptr10,int* __restrict__ ptr11,
                             int* __restrict__ ptr12,int* __restrict__ ptr13,int* __restrict__ ptr14,int* __restrict__ ptr15,
                             int* __restrict__ ptr16){
    svint32_t v1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t v2 = svld1_s32(svptrue_b32(),ptr2);
    svint32_t v3 = svld1_s32(svptrue_b32(),ptr3);
    svint32_t v4 = svld1_s32(svptrue_b32(),ptr4);
    svint32_t v5 = svld1_s32(svptrue_b32(),ptr5);
    svint32_t v6 = svld1_s32(svptrue_b32(),ptr6);
    svint32_t v7 = svld1_s32(svptrue_b32(),ptr7);
    svint32_t v8 = svld1_s32(svptrue_b32(),ptr8);
    svint32_t v9 = svld1_s32(svptrue_b32(),ptr9);
    svint32_t v10 = svld1_s32(svptrue_b32(),ptr10);
    svint32_t v11 = svld1_s32(svptrue_b32(),ptr11);
    svint32_t v12 = svld1_s32(svptrue_b32(),ptr12);
    svint32_t v13 = svld1_s32(svptrue_b32(),ptr13);
    svint32_t v14 = svld1_s32(svptrue_b32(),ptr14);
    svint32_t v15 = svld1_s32(svptrue_b32(),ptr15);
    svint32_t v16 = svld1_s32(svptrue_b32(),ptr16);
    CoreSmallSort16(v1,v2,v3,
                    v4,v5,v6,v7,
                    v8,v9,v10,v11,
                    v12,v13,v14,v15,
                    v16);
    svst1_s32(svptrue_b32(), ptr1, v1);
    svst1_s32(svptrue_b32(), ptr2, v2);
    svst1_s32(svptrue_b32(), ptr3, v3);
    svst1_s32(svptrue_b32(), ptr4, v4);
    svst1_s32(svptrue_b32(), ptr5, v5);
    svst1_s32(svptrue_b32(), ptr6, v6);
    svst1_s32(svptrue_b32(), ptr7, v7);
    svst1_s32(svptrue_b32(), ptr8, v8);
    svst1_s32(svptrue_b32(), ptr9, v9);
    svst1_s32(svptrue_b32(), ptr10, v10);
    svst1_s32(svptrue_b32(), ptr11, v11);
    svst1_s32(svptrue_b32(), ptr12, v12);
    svst1_s32(svptrue_b32(), ptr13, v13);
    svst1_s32(svptrue_b32(), ptr14, v14);
    svst1_s32(svptrue_b32(), ptr15, v15);
    svst1_s32(svptrue_b32(), ptr16, v16);
}


inline void CoreSmallSort16( double* __restrict__ ptr1,double* __restrict__ ptr2,double* __restrict__ ptr3,
                             double* __restrict__ ptr4,double* __restrict__ ptr5,double* __restrict__ ptr6,double* __restrict__ ptr7,
                             double* __restrict__ ptr8,double* __restrict__ ptr9,double* __restrict__ ptr10,double* __restrict__ ptr11,
                             double* __restrict__ ptr12,double* __restrict__ ptr13,double* __restrict__ ptr14,double* __restrict__ ptr15,
                             double* __restrict__ ptr16){
    svfloat64_t v1 = svld1_f64(svptrue_b64(),ptr1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(),ptr2);
    svfloat64_t v3 = svld1_f64(svptrue_b64(),ptr3);
    svfloat64_t v4 = svld1_f64(svptrue_b64(),ptr4);
    svfloat64_t v5 = svld1_f64(svptrue_b64(),ptr5);
    svfloat64_t v6 = svld1_f64(svptrue_b64(),ptr6);
    svfloat64_t v7 = svld1_f64(svptrue_b64(),ptr7);
    svfloat64_t v8 = svld1_f64(svptrue_b64(),ptr8);
    svfloat64_t v9 = svld1_f64(svptrue_b64(),ptr9);
    svfloat64_t v10 = svld1_f64(svptrue_b64(),ptr10);
    svfloat64_t v11 = svld1_f64(svptrue_b64(),ptr11);
    svfloat64_t v12 = svld1_f64(svptrue_b64(),ptr12);
    svfloat64_t v13 = svld1_f64(svptrue_b64(),ptr13);
    svfloat64_t v14 = svld1_f64(svptrue_b64(),ptr14);
    svfloat64_t v15 = svld1_f64(svptrue_b64(),ptr15);
    svfloat64_t v16 = svld1_f64(svptrue_b64(),ptr16);
    CoreSmallSort16(v1,v2,v3,
                    v4,v5,v6,v7,
                    v8,v9,v10,v11,
                    v12,v13,v14,v15,
                    v16);
    svst1_f64(svptrue_b64(), ptr1, v1);
    svst1_f64(svptrue_b64(), ptr2, v2);
    svst1_f64(svptrue_b64(), ptr3, v3);
    svst1_f64(svptrue_b64(), ptr4, v4);
    svst1_f64(svptrue_b64(), ptr5, v5);
    svst1_f64(svptrue_b64(), ptr6, v6);
    svst1_f64(svptrue_b64(), ptr7, v7);
    svst1_f64(svptrue_b64(), ptr8, v8);
    svst1_f64(svptrue_b64(), ptr9, v9);
    svst1_f64(svptrue_b64(), ptr10, v10);
    svst1_f64(svptrue_b64(), ptr11, v11);
    svst1_f64(svptrue_b64(), ptr12, v12);
    svst1_f64(svptrue_b64(), ptr13, v13);
    svst1_f64(svptrue_b64(), ptr14, v14);
    svst1_f64(svptrue_b64(), ptr15, v15);
    svst1_f64(svptrue_b64(), ptr16, v16);
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
        if(part && left < part-1)  CoreSort<SortType,IndexType>(array,left,part - 1);
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
