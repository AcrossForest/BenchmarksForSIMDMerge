//////////////////////////////////////////////////////////
/// By berenger.bramas@inria.fr 2020.
/// Licence is MIT.
/// Comes without any warranty.
///
/// Code to sort two arrays of integers (key/value)
/// using ARM SVE (but works only for vectors of 512 bits).
/// It also includes a partitioning function.
///
/// Please refer to the README to know how to build
/// and to have more information about the functions.
///
//////////////////////////////////////////////////////////
#ifndef SORTSVEKV512_HPP
#define SORTSVEKV512_HPP

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

namespace SortSVEkv512 {

inline void CoreSmallSort(svint32_t& input, svint32_t& values){
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
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);

        values = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), values, svtbl_s32( values, idxNoNeigh));

        input = tmp_input;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1213141589101145670123);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t tmp_input  = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);

        values = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), values, svtbl_s32( values, idxNoNeigh));

        input = tmp_input;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t tmp_input  = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);

        values = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), values, svtbl_s32( values, idxNoNeigh));

        input = tmp_input;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array8910111213141501234567);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t tmp_input  = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);

        values = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), values, svtbl_s32( values, idxNoNeigh));

        input = tmp_input;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t tmp_input  = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);

        values = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), values, svtbl_s32( values, idxNoNeigh));

        input = tmp_input;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t tmp_input  = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);

        values = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), values, svtbl_s32( values, idxNoNeigh));

        input = tmp_input;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t tmp_input  = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);

        values = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), values, svtbl_s32( values, idxNoNeigh));

        input = tmp_input;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t tmp_input  = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);

        values = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), values, svtbl_s32( values, idxNoNeigh));

        input = tmp_input;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t tmp_input  = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);

        values = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), values, svtbl_s32( values, idxNoNeigh));

        input = tmp_input;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t tmp_input  = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);

        values = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), values, svtbl_s32( values, idxNoNeigh));

        input = tmp_input;
    }
}

inline void CoreSmallSort(int* __restrict__ ptr1, int* __restrict__ ptrVal){
    svint32_t v = svld1_s32(svptrue_b32(),ptr1);
    svint32_t v_val = svld1_s32(svptrue_b32(),ptrVal);
    CoreSmallSort(v, v_val);
    svst1_s32(svptrue_b32(),ptr1, v);
    svst1_s32(svptrue_b32(),ptrVal, v_val);
}



inline void CoreExchangeSort2V(svint32_t& input, svint32_t& input2,
                                 svint32_t& input_val, svint32_t& input2_val){
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
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input2, permNeigh);
        svint32_t tmp_input2 = svmax_s32_z(svptrue_b32(),input2, permNeigh);

        svint32_t input_val_perm = svtbl_s32( input_val, idxNoNeigh);
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, permNeigh), input_val_perm, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input_val_perm);

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
         svint32_t tmp_input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
         svint32_t tmp_input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);

         input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
         input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

         input = tmp_input;
         input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
         svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
         svint32_t tmp_input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);

         input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
         input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

         input = tmp_input;
         input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
         svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
         svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);

         input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
         input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

         input = tmp_input;
         input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
         svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
         svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);

         input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
         input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

         input = tmp_input;
         input2 = tmp_input2;
    }
}



inline void CoreSmallSort2(svint32_t& input, svint32_t& input2,
                                 svint32_t& input_val, svint32_t& input2_val){
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
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1213141589101145670123);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array8910111213141501234567);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t tmp_input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
    }
    CoreExchangeSort2V(input,input2,input_val,input2_val);
}


inline void CoreSmallSort2(int* __restrict__ ptr1, int* __restrict__ values ){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+16);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+16);
    CoreSmallSort2(input1, input2, input1_val, input2_val);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr1+16, input2);
    svst1_s32(svptrue_b32(),values, input1_val);
    svst1_s32(svptrue_b32(),values+16, input2_val);
}


inline void CoreSmallSort3(svint32_t& input, svint32_t& input2, svint32_t& input3,
                                 svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val){
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
    CoreSmallSort2(input, input2, input_val, input2_val);
    CoreSmallSort(input3, input3_val);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh = svtbl_s32( input3, idxNoNeigh);
        svint32_t tmp_input3 = svmax_s32_z(svptrue_b32(),input2, permNeigh);
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input2, permNeigh);

        svint32_t input3_val_perm = svtbl_s32( input3_val, idxNoNeigh);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, permNeigh), input3_val_perm, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input3_val_perm);

        input3 = tmp_input3;
        input2 = tmp_input2;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t tmp_input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);

        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input_val_copy);

        input = tmp_input;
        input2 = tmp_input2;
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
        svint32_t tmp_input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
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
        svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
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
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
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
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
    }
}


inline void CoreSmallSort3(int* __restrict__ ptr1, int* __restrict__ values){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+16);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+32);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+16);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+32);
    CoreSmallSort3(input1, input2, input3,
                         input1_val, input2_val, input3_val);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr1+16, input2);
    svst1_s32(svptrue_b32(),ptr1+32, input3);
    svst1_s32(svptrue_b32(),values, input1_val);
    svst1_s32(svptrue_b32(),values+16, input2_val);
    svst1_s32(svptrue_b32(),values+32, input3_val);
}



inline void CoreSmallSort4(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                                 svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val){
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
    CoreSmallSort2(input, input2, input_val, input2_val);
    CoreSmallSort2(input3, input4, input3_val, input4_val);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh3 = svtbl_s32( input3, idxNoNeigh);
        svint32_t permNeigh4 = svtbl_s32( input4, idxNoNeigh);

        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input, permNeigh4);
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input, permNeigh4);

        svint32_t tmp_input3 = svmax_s32_z(svptrue_b32(),input2, permNeigh3);
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input2, permNeigh3);


        svint32_t input4_val_perm = svtbl_s32( input4_val, idxNoNeigh);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, permNeigh4), input4_val_perm, input_val);
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, input4_val_perm);

        svint32_t input3_val_perm = svtbl_s32( input3_val, idxNoNeigh);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, permNeigh3), input3_val_perm, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input3_val_perm);

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t tmp_input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);

        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input_val_copy);

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svint32_t inputCopy = input3;
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);

        svint32_t input3_val_copy = input3_val;
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, inputCopy), input3_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input3_val_copy);

        input3 = tmp_input3;
        input4 = tmp_input4;
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
        svint32_t tmp_input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
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
        svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
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
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
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
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
    }
}


inline void CoreSmallSort4(int* __restrict__ ptr1, int* __restrict__ values ){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+16);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+32);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+48);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+16);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+32);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+48);
    CoreSmallSort4(input1, input2, input3, input4,
                         input1_val, input2_val, input3_val, input4_val);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr1+16, input2);
    svst1_s32(svptrue_b32(),ptr1+32, input3);
    svst1_s32(svptrue_b32(),ptr1+48, input4);
    svst1_s32(svptrue_b32(),values, input1_val);
    svst1_s32(svptrue_b32(),values+16, input2_val);
    svst1_s32(svptrue_b32(),values+32, input3_val);
    svst1_s32(svptrue_b32(),values+48, input4_val);
}


inline void CoreSmallSort5(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4, svint32_t& input5,
                                 svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val, svint32_t& input5_val){
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
    CoreSmallSort4(input, input2, input3, input4,
                         input_val, input2_val, input3_val, input4_val);
    CoreSmallSort(input5, input5_val);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);

        svint32_t tmp_input5 = svmax_s32_z(svptrue_b32(),input4, permNeigh5);
        svint32_t tmp_input4 = svmin_s32_z(svptrue_b32(),input4, permNeigh5);

        svint32_t input5_val_copy = svtbl_s32( input5_val, idxNoNeigh);
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, permNeigh5), input5_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input5_val_copy);

        input5 = tmp_input5;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t tmp_input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);

        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input2_val);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, input_val_copy);

        input = tmp_input;
        input3 = tmp_input3;
    }
    {
        svint32_t inputCopy = input2;
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);

        svint32_t input2_val_copy = input2_val;
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, inputCopy), input2_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input2_val_copy);

        input2 = tmp_input2;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t tmp_input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);

        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input_val_copy);

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svint32_t inputCopy = input3;
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);

        svint32_t input3_val_copy = input3_val;
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, inputCopy), input3_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input3_val_copy);

        input3 = tmp_input3;
        input4 = tmp_input4;
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
        svint32_t tmp_input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xFF00,  permNeighMax5, permNeighMin5);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
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
        svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xF0F0,  permNeighMax5, permNeighMin5);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
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
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xCCCC,  permNeighMax5, permNeighMin5);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
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
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xAAAA,  permNeighMax5, permNeighMin5);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
    }
}


inline void CoreSmallSort5(int* __restrict__ ptr1, int* __restrict__ values){
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+1*16);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+2*16);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+3*16);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+4*16);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+1*16);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+2*16);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+3*16);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+4*16);
    CoreSmallSort5(input1, input2, input3, input4, input5,
                    input1_val, input2_val, input3_val, input4_val, input5_val);
    svst1_s32(svptrue_b32(),ptr1, input1);
    svst1_s32(svptrue_b32(),ptr1+1*16, input2);
    svst1_s32(svptrue_b32(),ptr1+2*16, input3);
    svst1_s32(svptrue_b32(),ptr1+3*16, input4);
    svst1_s32(svptrue_b32(),ptr1+4*16, input5);
    svst1_s32(svptrue_b32(),values, input1_val);
    svst1_s32(svptrue_b32(),values+1*16, input2_val);
    svst1_s32(svptrue_b32(),values+2*16, input3_val);
    svst1_s32(svptrue_b32(),values+3*16, input4_val);
    svst1_s32(svptrue_b32(),values+4*16, input5_val);
}



inline void CoreSmallSort6(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6,
                                 svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                                                             svint32_t& input5_val, svint32_t& input6_val){
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
    CoreSmallSort4(input, input2, input3, input4,
                         input_val, input2_val, input3_val, input4_val);
    CoreSmallSort2(input5, input6, input5_val, input6_val);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);

        svint32_t tmp_input5 = svmax_s32_z(svptrue_b32(),input4, permNeigh5);
        svint32_t tmp_input4 = svmin_s32_z(svptrue_b32(),input4, permNeigh5);

        svint32_t tmp_input6 = svmax_s32_z(svptrue_b32(),input3, permNeigh6);
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input3, permNeigh6);


        svint32_t input5_val_perm = svtbl_s32( input5_val, idxNoNeigh);
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, permNeigh5), input5_val_perm, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input5_val_perm);

        svint32_t input6_val_perm = svtbl_s32( input6_val, idxNoNeigh);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, permNeigh6), input6_val_perm, input3_val);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, input6_val_perm);

        input5 = tmp_input5;
        input4 = tmp_input4;

        input6 = tmp_input6;
        input3 = tmp_input3;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t tmp_input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input3_val);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, input_val_copy);

        input = tmp_input;
        input3 = tmp_input3;
    }
    {
        svint32_t inputCopy = input2;
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t input2_val_copy = input2_val;
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, inputCopy), input2_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input2_val_copy);

        input2 = tmp_input2;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t tmp_input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input_val_copy);

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svint32_t inputCopy = input3;
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t input3_val_copy = input3_val;
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, inputCopy), input3_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input3_val_copy);

        input3 = tmp_input3;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input5;
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t tmp_input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t input5_val_copy = input5_val;
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, inputCopy), input5_val_copy, input6_val);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, input5_val_copy);

        input5 = tmp_input5;
        input6 = tmp_input6;
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
        svint32_t tmp_input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xFF00,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xFF00,  permNeighMax6, permNeighMin6);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
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
        svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xF0F0,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xF0F0,  permNeighMax6, permNeighMin6);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
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
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xCCCC,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xCCCC,  permNeighMax6, permNeighMin6);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
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
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xAAAA,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xAAAA,  permNeighMax6, permNeighMin6);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
    }
}


inline void CoreSmallSort6(int* __restrict__ ptr1, int* __restrict__ values ){
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1+0*16);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+1*16);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*16);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*16);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*16);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*16);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values+0*16);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+1*16);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*16);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*16);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*16);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*16);
    CoreSmallSort6(input0,input1,input2,input3,input4,input5,
        input0_val,input1_val,input2_val,input3_val,input4_val,input5_val);
    svst1_s32(svptrue_b32(),ptr1+0*16, input0);
    svst1_s32(svptrue_b32(),ptr1+1*16, input1);
    svst1_s32(svptrue_b32(),ptr1+2*16, input2);
    svst1_s32(svptrue_b32(),ptr1+3*16, input3);
    svst1_s32(svptrue_b32(),ptr1+4*16, input4);
    svst1_s32(svptrue_b32(),ptr1+5*16, input5);
    svst1_s32(svptrue_b32(),values+0*16, input0_val);
    svst1_s32(svptrue_b32(),values+1*16, input1_val);
    svst1_s32(svptrue_b32(),values+2*16, input2_val);
    svst1_s32(svptrue_b32(),values+3*16, input3_val);
    svst1_s32(svptrue_b32(),values+4*16, input4_val);
    svst1_s32(svptrue_b32(),values+5*16, input5_val);
}



inline void CoreSmallSort7(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7,
                                 svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                                svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val){
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
    CoreSmallSort4(input, input2, input3, input4,
                         input_val, input2_val, input3_val, input4_val);
    CoreSmallSort3(input5, input6, input7,
                         input5_val, input6_val, input7_val);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);

        svint32_t tmp_input5 = svmax_s32_z(svptrue_b32(),input4, permNeigh5);
        svint32_t tmp_input4 = svmin_s32_z(svptrue_b32(),input4, permNeigh5);

        svint32_t tmp_input6 = svmax_s32_z(svptrue_b32(),input3, permNeigh6);
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input3, permNeigh6);

        svint32_t tmp_input7 = svmax_s32_z(svptrue_b32(),input2, permNeigh7);
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input2, permNeigh7);


        svint32_t input5_val_perm = svtbl_s32( input5_val, idxNoNeigh);
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, permNeigh5), input5_val_perm, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input5_val_perm);

        svint32_t input6_val_perm = svtbl_s32( input6_val, idxNoNeigh);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, permNeigh6), input6_val_perm, input3_val);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, input6_val_perm);

        svint32_t input7_val_perm = svtbl_s32( input7_val, idxNoNeigh);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, permNeigh7), input7_val_perm, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input7_val_perm);


        input5 = tmp_input5;
        input4 = tmp_input4;

        input6 = tmp_input6;
        input3 = tmp_input3;

        input7 = tmp_input7;
        input2 = tmp_input2;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t tmp_input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input3_val);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, input_val_copy);

        input = tmp_input;
        input3 = tmp_input3;
    }
    {
        svint32_t inputCopy = input2;
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t input2_val_copy = input2_val;
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, inputCopy), input2_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input2_val_copy);

        input2 = tmp_input2;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t tmp_input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input_val_copy);

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svint32_t inputCopy = input3;
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t input3_val_copy = input3_val;
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, inputCopy), input3_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input3_val_copy);

        input3 = tmp_input3;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input5;
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input7, inputCopy);
        svint32_t tmp_input7 = svmax_s32_z(svptrue_b32(),input7, inputCopy);
        svint32_t input5_val_copy = input5_val;
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, inputCopy), input5_val_copy, input7_val);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, input5_val_copy);

        input5 = tmp_input5;
        input7 = tmp_input7;
    }
    {
        svint32_t inputCopy = input5;
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t tmp_input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t input5_val_copy = input5_val;
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, inputCopy), input5_val_copy, input6_val);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, input5_val_copy);

        input5 = tmp_input5;
        input6 = tmp_input6;
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
        svint32_t tmp_input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xFF00,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xFF00,  permNeighMax6, permNeighMin6);
        svint32_t tmp_input7 = svsel_s32(mask0xFF00,  permNeighMax7, permNeighMin7);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, svtbl_s32( input7_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
        input7 = tmp_input7;
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
        svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xF0F0,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xF0F0,  permNeighMax6, permNeighMin6);
        svint32_t tmp_input7 = svsel_s32(mask0xF0F0,  permNeighMax7, permNeighMin7);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, svtbl_s32( input7_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
        input7 = tmp_input7;
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
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xCCCC,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xCCCC,  permNeighMax6, permNeighMin6);
        svint32_t tmp_input7 = svsel_s32(mask0xCCCC,  permNeighMax7, permNeighMin7);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, svtbl_s32( input7_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
        input7 = tmp_input7;
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
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xAAAA,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xAAAA,  permNeighMax6, permNeighMin6);
        svint32_t tmp_input7 = svsel_s32(mask0xAAAA,  permNeighMax7, permNeighMin7);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, svtbl_s32( input7_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
        input7 = tmp_input7;
    }
}


inline void CoreSmallSort7(int* __restrict__ ptr1, int* __restrict__ values ){
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1+0*16);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+1*16);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*16);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*16);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*16);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*16);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*16);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values+0*16);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+1*16);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*16);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*16);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*16);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*16);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*16);
    CoreSmallSort7(input0,input1,input2,input3,input4,input5,input6,
        input0_val,input1_val,input2_val,input3_val,input4_val,input5_val,input6_val);
    svst1_s32(svptrue_b32(),ptr1+0*16, input0);
    svst1_s32(svptrue_b32(),ptr1+1*16, input1);
    svst1_s32(svptrue_b32(),ptr1+2*16, input2);
    svst1_s32(svptrue_b32(),ptr1+3*16, input3);
    svst1_s32(svptrue_b32(),ptr1+4*16, input4);
    svst1_s32(svptrue_b32(),ptr1+5*16, input5);
    svst1_s32(svptrue_b32(),ptr1+6*16, input6);
    svst1_s32(svptrue_b32(),values+0*16, input0_val);
    svst1_s32(svptrue_b32(),values+1*16, input1_val);
    svst1_s32(svptrue_b32(),values+2*16, input2_val);
    svst1_s32(svptrue_b32(),values+3*16, input3_val);
    svst1_s32(svptrue_b32(),values+4*16, input4_val);
    svst1_s32(svptrue_b32(),values+5*16, input5_val);
    svst1_s32(svptrue_b32(),values+6*16, input6_val);
}



inline void CoreSmallSort8(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                                 svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                                 svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val){
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
    CoreSmallSort4(input, input2, input3, input4,
                         input_val, input2_val, input3_val, input4_val);
    CoreSmallSort4(input5, input6, input7, input8,
                         input5_val, input6_val, input7_val, input8_val);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh5 = svtbl_s32( input5, idxNoNeigh);
        svint32_t permNeigh6 = svtbl_s32( input6, idxNoNeigh);
        svint32_t permNeigh7 = svtbl_s32( input7, idxNoNeigh);
        svint32_t permNeigh8 = svtbl_s32( input8, idxNoNeigh);

        svint32_t tmp_input5 = svmax_s32_z(svptrue_b32(),input4, permNeigh5);
        svint32_t tmp_input4 = svmin_s32_z(svptrue_b32(),input4, permNeigh5);

        svint32_t tmp_input6 = svmax_s32_z(svptrue_b32(),input3, permNeigh6);
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input3, permNeigh6);

        svint32_t tmp_input7 = svmax_s32_z(svptrue_b32(),input2, permNeigh7);
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input2, permNeigh7);

        svint32_t tmp_input8 = svmax_s32_z(svptrue_b32(),input, permNeigh8);
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input, permNeigh8);


        svint32_t input5_val_perm = svtbl_s32( input5_val, idxNoNeigh);
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, permNeigh5), input5_val_perm, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input5_val_perm);

        svint32_t input6_val_perm = svtbl_s32( input6_val, idxNoNeigh);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, permNeigh6), input6_val_perm, input3_val);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, input6_val_perm);

        svint32_t input7_val_perm = svtbl_s32( input7_val, idxNoNeigh);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, permNeigh7), input7_val_perm, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input7_val_perm);

        svint32_t input8_val_perm = svtbl_s32( input8_val, idxNoNeigh);
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, permNeigh8), input8_val_perm, input_val);
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, input8_val_perm);


        input5 = tmp_input5;
        input4 = tmp_input4;

        input6 = tmp_input6;
        input3 = tmp_input3;

        input7 = tmp_input7;
        input2 = tmp_input2;

        input8 = tmp_input8;
        input = tmp_input;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t tmp_input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input3_val);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, input_val_copy);

        input = tmp_input;
        input3 = tmp_input3;
    }
    {
        svint32_t inputCopy = input2;
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t input2_val_copy = input2_val;
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, inputCopy), input2_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input2_val_copy);

        input2 = tmp_input2;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t tmp_input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input_val_copy);

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svint32_t inputCopy = input3;
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t input3_val_copy = input3_val;
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, inputCopy), input3_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input3_val_copy);

        input3 = tmp_input3;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input5;
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input7, inputCopy);
        svint32_t tmp_input7 = svmax_s32_z(svptrue_b32(),input7, inputCopy);
        svint32_t input5_val_copy = input5_val;
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, inputCopy), input5_val_copy, input7_val);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, input5_val_copy);

        input5 = tmp_input5;
        input7 = tmp_input7;
    }
    {
        svint32_t inputCopy = input6;
        svint32_t tmp_input6 = svmin_s32_z(svptrue_b32(),input8, inputCopy);
        svint32_t tmp_input8 = svmax_s32_z(svptrue_b32(),input8, inputCopy);
        svint32_t input6_val_copy = input6_val;
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, inputCopy), input6_val_copy, input8_val);
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, input6_val_copy);

        input6 = tmp_input6;
        input8 = tmp_input8;
    }
    {
        svint32_t inputCopy = input5;
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t tmp_input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t input5_val_copy = input5_val;
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, inputCopy), input5_val_copy, input6_val);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, input5_val_copy);

        input5 = tmp_input5;
        input6 = tmp_input6;
    }
    {
        svint32_t inputCopy = input7;
        svint32_t tmp_input7 = svmin_s32_z(svptrue_b32(),input8, inputCopy);
        svint32_t tmp_input8 = svmax_s32_z(svptrue_b32(),input8, inputCopy);
        svint32_t input7_val_copy = input7_val;
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, inputCopy), input7_val_copy, input8_val);
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, input7_val_copy);

        input7 = tmp_input7;
        input8 = tmp_input8;
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
        svint32_t tmp_input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xFF00,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xFF00,  permNeighMax6, permNeighMin6);
        svint32_t tmp_input7 = svsel_s32(mask0xFF00,  permNeighMax7, permNeighMin7);
        svint32_t tmp_input8 = svsel_s32(mask0xFF00,  permNeighMax8, permNeighMin8);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, svtbl_s32( input7_val, idxNoNeigh));
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, svtbl_s32( input8_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
        input7 = tmp_input7;
        input8 = tmp_input8;
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
        svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xF0F0,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xF0F0,  permNeighMax6, permNeighMin6);
        svint32_t tmp_input7 = svsel_s32(mask0xF0F0,  permNeighMax7, permNeighMin7);
        svint32_t tmp_input8 = svsel_s32(mask0xF0F0,  permNeighMax8, permNeighMin8);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, svtbl_s32( input7_val, idxNoNeigh));
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, svtbl_s32( input8_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
        input7 = tmp_input7;
        input8 = tmp_input8;
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
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xCCCC,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xCCCC,  permNeighMax6, permNeighMin6);
        svint32_t tmp_input7 = svsel_s32(mask0xCCCC,  permNeighMax7, permNeighMin7);
        svint32_t tmp_input8 = svsel_s32(mask0xCCCC,  permNeighMax8, permNeighMin8);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, svtbl_s32( input7_val, idxNoNeigh));
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, svtbl_s32( input8_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
        input7 = tmp_input7;
        input8 = tmp_input8;
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
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xAAAA,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xAAAA,  permNeighMax6, permNeighMin6);
        svint32_t tmp_input7 = svsel_s32(mask0xAAAA,  permNeighMax7, permNeighMin7);
        svint32_t tmp_input8 = svsel_s32(mask0xAAAA,  permNeighMax8, permNeighMin8);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, svtbl_s32( input7_val, idxNoNeigh));
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, svtbl_s32( input8_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
        input7 = tmp_input7;
        input8 = tmp_input8;
    }
}

inline void CoreSmallSort8(int* __restrict__ ptr1, int* __restrict__ values ){
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1+0*16);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+1*16);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*16);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*16);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*16);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*16);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*16);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*16);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values+0*16);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+1*16);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*16);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*16);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*16);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*16);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*16);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*16);
    CoreSmallSort8(input0,input1,input2,input3,input4,input5,input6,input7,
        input0_val,input1_val,input2_val,input3_val,input4_val,input5_val,input6_val,input7_val);
    svst1_s32(svptrue_b32(),ptr1+0*16, input0);
    svst1_s32(svptrue_b32(),ptr1+1*16, input1);
    svst1_s32(svptrue_b32(),ptr1+2*16, input2);
    svst1_s32(svptrue_b32(),ptr1+3*16, input3);
    svst1_s32(svptrue_b32(),ptr1+4*16, input4);
    svst1_s32(svptrue_b32(),ptr1+5*16, input5);
    svst1_s32(svptrue_b32(),ptr1+6*16, input6);
    svst1_s32(svptrue_b32(),ptr1+7*16, input7);
    svst1_s32(svptrue_b32(),values+0*16, input0_val);
    svst1_s32(svptrue_b32(),values+1*16, input1_val);
    svst1_s32(svptrue_b32(),values+2*16, input2_val);
    svst1_s32(svptrue_b32(),values+3*16, input3_val);
    svst1_s32(svptrue_b32(),values+4*16, input4_val);
    svst1_s32(svptrue_b32(),values+5*16, input5_val);
    svst1_s32(svptrue_b32(),values+6*16, input6_val);
    svst1_s32(svptrue_b32(),values+7*16, input7_val);
}



inline void CoreSmallEnd1(svint32_t& input, svint32_t& values){
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
        svint32_t tmp_input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);

        values = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), values, svtbl_s32( values, idxNoNeigh));

        input = tmp_input;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);

        values = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), values, svtbl_s32( values, idxNoNeigh));

        input = tmp_input;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);

        values = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), values, svtbl_s32( values, idxNoNeigh));

        input = tmp_input;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);

        values = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), values, svtbl_s32( values, idxNoNeigh));

        input = tmp_input;
    }
}

inline void CoreSmallEnd2(svint32_t& input, svint32_t& input2,
                                   svint32_t& input_val, svint32_t& input2_val){
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
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t tmp_input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input_val_copy);

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array7654321015141312111098);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t tmp_input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1110981514131232107654);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1312151498111054761032);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array1415121310118967452301);
        svint32_t permNeigh = svtbl_s32( input, idxNoNeigh);
        svint32_t permNeigh2 = svtbl_s32( input2, idxNoNeigh);
        svint32_t permNeighMin = svmin_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMin2 = svmin_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t permNeighMax = svmax_s32_z(svptrue_b32(),permNeigh, input);
        svint32_t permNeighMax2 = svmax_s32_z(svptrue_b32(),permNeigh2, input2);
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
    }
}

inline void CoreSmallEnd3(svint32_t& input, svint32_t& input2, svint32_t& input3,
                                   svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val){
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
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t tmp_input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input3_val);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, input_val_copy);

        input = tmp_input;
        input3 = tmp_input3;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t tmp_input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input_val_copy);

        input = tmp_input;
        input2 = tmp_input2;
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
        svint32_t tmp_input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
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
        svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
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
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
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
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
    }
}

inline void CoreSmallEnd4(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                                   svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val){
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
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t tmp_input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input3_val);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, input_val_copy);

        input = tmp_input;
        input3 = tmp_input3;
    }
    {
        svint32_t inputCopy = input2;
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t input2_val_copy = input2_val;
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, inputCopy), input2_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input2_val_copy);

        input2 = tmp_input2;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t tmp_input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input_val_copy);

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svint32_t inputCopy = input3;
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t input3_val_copy = input3_val;
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, inputCopy), input3_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input3_val_copy);

        input3 = tmp_input3;
        input4 = tmp_input4;
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
        svint32_t tmp_input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
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
        svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
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
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
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
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
    }
}

inline void CoreSmallEnd5(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5,
                                   svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                                   svint32_t& input5_val){
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
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input5, inputCopy);
        svint32_t tmp_input5 = svmax_s32_z(svptrue_b32(),input5, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input5_val);
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, input_val_copy);

        input = tmp_input;
        input5 = tmp_input5;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t tmp_input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input3_val);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, input_val_copy);

        input = tmp_input;
        input3 = tmp_input3;
    }
    {
        svint32_t inputCopy = input2;
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t input2_val_copy = input2_val;
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, inputCopy), input2_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input2_val_copy);

        input2 = tmp_input2;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t tmp_input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input_val_copy);

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svint32_t inputCopy = input3;
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t input3_val_copy = input3_val;
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, inputCopy), input3_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input3_val_copy);

        input3 = tmp_input3;
        input4 = tmp_input4;
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
        svint32_t tmp_input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xFF00,  permNeighMax5, permNeighMin5);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
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
        svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xF0F0,  permNeighMax5, permNeighMin5);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
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
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xCCCC,  permNeighMax5, permNeighMin5);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
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
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xAAAA,  permNeighMax5, permNeighMin5);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
    }
}

inline void CoreSmallEnd6(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6,
                                   svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                                   svint32_t& input5_val, svint32_t& input6_val){
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
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input5, inputCopy);
        svint32_t tmp_input5 = svmax_s32_z(svptrue_b32(),input5, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input5_val);
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, input_val_copy);

        input = tmp_input;
        input5 = tmp_input5;
    }
    {
        svint32_t inputCopy = input2;
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t tmp_input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t input2_val_copy = input2_val;
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, inputCopy), input2_val_copy, input6_val);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, input2_val_copy);

        input2 = tmp_input2;
        input6 = tmp_input6;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t tmp_input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input3_val);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, input_val_copy);

        input = tmp_input;
        input3 = tmp_input3;
    }
    {
        svint32_t inputCopy = input2;
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t input2_val_copy = input2_val;
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, inputCopy), input2_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input2_val_copy);

        input2 = tmp_input2;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t tmp_input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input_val_copy);

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svint32_t inputCopy = input3;
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t input3_val_copy = input3_val;
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, inputCopy), input3_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input3_val_copy);

        input3 = tmp_input3;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input5;
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t tmp_input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t input5_val_copy = input5_val;
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, inputCopy), input5_val_copy, input6_val);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, input5_val_copy);

        input5 = tmp_input5;
        input6 = tmp_input6;
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
        svint32_t tmp_input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xFF00,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xFF00,  permNeighMax6, permNeighMin6);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
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
        svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xF0F0,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xF0F0,  permNeighMax6, permNeighMin6);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
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
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xCCCC,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xCCCC,  permNeighMax6, permNeighMin6);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
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
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xAAAA,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xAAAA,  permNeighMax6, permNeighMin6);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
    }
}



inline void CoreSmallEnd7(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7,
                                   svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                                   svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val){
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
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input5, inputCopy);
        svint32_t tmp_input5 = svmax_s32_z(svptrue_b32(),input5, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input5_val);
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, input_val_copy);

        input = tmp_input;
        input5 = tmp_input5;
    }
    {
        svint32_t inputCopy = input2;
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t tmp_input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t input2_val_copy = input2_val;
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, inputCopy), input2_val_copy, input6_val);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, input2_val_copy);

        input2 = tmp_input2;
        input6 = tmp_input6;
    }
    {
        svint32_t inputCopy = input3;
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input7, inputCopy);
        svint32_t tmp_input7 = svmax_s32_z(svptrue_b32(),input7, inputCopy);
        svint32_t input3_val_copy = input3_val;
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, inputCopy), input3_val_copy, input7_val);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, input3_val_copy);

        input3 = tmp_input3;
        input7 = tmp_input7;
    }
    {
        svint32_t inputCopy = input5;
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input7, inputCopy);
        svint32_t tmp_input7 = svmax_s32_z(svptrue_b32(),input7, inputCopy);
        svint32_t input5_val_copy = input5_val;
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, inputCopy), input5_val_copy, input7_val);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, input5_val_copy);

        input5 = tmp_input5;
        input7 = tmp_input7;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t tmp_input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input3_val);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, input_val_copy);

        input = tmp_input;
        input3 = tmp_input3;
    }
    {
        svint32_t inputCopy = input2;
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t input2_val_copy = input2_val;
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, inputCopy), input2_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input2_val_copy);

        input2 = tmp_input2;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t tmp_input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input_val_copy);

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svint32_t inputCopy = input3;
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t input3_val_copy = input3_val;
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, inputCopy), input3_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input3_val_copy);

        input3 = tmp_input3;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input5;
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input7, inputCopy);
        svint32_t tmp_input7 = svmax_s32_z(svptrue_b32(),input7, inputCopy);
        svint32_t input5_val_copy = input5_val;
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, inputCopy), input5_val_copy, input7_val);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, input5_val_copy);

        input5 = tmp_input5;
        input7 = tmp_input7;
    }
    {
        svint32_t inputCopy = input5;
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t tmp_input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t input5_val_copy = input5_val;
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, inputCopy), input5_val_copy, input6_val);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, input5_val_copy);

        input5 = tmp_input5;
        input6 = tmp_input6;
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
        svint32_t tmp_input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xFF00,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xFF00,  permNeighMax6, permNeighMin6);
        svint32_t tmp_input7 = svsel_s32(mask0xFF00,  permNeighMax7, permNeighMin7);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, svtbl_s32( input7_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
        input7 = tmp_input7;
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
        svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xF0F0,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xF0F0,  permNeighMax6, permNeighMin6);
        svint32_t tmp_input7 = svsel_s32(mask0xF0F0,  permNeighMax7, permNeighMin7);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, svtbl_s32( input7_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
        input7 = tmp_input7;
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
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xCCCC,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xCCCC,  permNeighMax6, permNeighMin6);
        svint32_t tmp_input7 = svsel_s32(mask0xCCCC,  permNeighMax7, permNeighMin7);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, svtbl_s32( input7_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
        input7 = tmp_input7;
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
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xAAAA,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xAAAA,  permNeighMax6, permNeighMin6);
        svint32_t tmp_input7 = svsel_s32(mask0xAAAA,  permNeighMax7, permNeighMin7);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, svtbl_s32( input7_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
        input7 = tmp_input7;
    }
}



inline void CoreSmallEnd8(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                                   svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                                   svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val){
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
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input5, inputCopy);
        svint32_t tmp_input5 = svmax_s32_z(svptrue_b32(),input5, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input5_val);
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, input_val_copy);

        input = tmp_input;
        input5 = tmp_input5;
    }
    {
        svint32_t inputCopy = input2;
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t tmp_input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t input2_val_copy = input2_val;
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, inputCopy), input2_val_copy, input6_val);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, input2_val_copy);

        input2 = tmp_input2;
        input6 = tmp_input6;
    }
    {
        svint32_t inputCopy = input3;
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input7, inputCopy);
        svint32_t tmp_input7 = svmax_s32_z(svptrue_b32(),input7, inputCopy);
        svint32_t input3_val_copy = input3_val;
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, inputCopy), input3_val_copy, input7_val);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, input3_val_copy);

        input3 = tmp_input3;
        input7 = tmp_input7;
    }
    {
        svint32_t inputCopy = input4;
        svint32_t tmp_input4 = svmin_s32_z(svptrue_b32(),input8, inputCopy);
        svint32_t tmp_input8 = svmax_s32_z(svptrue_b32(),input8, inputCopy);
        svint32_t input4_val_copy = input4_val;
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, inputCopy), input4_val_copy, input8_val);
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, input4_val_copy);

        input4 = tmp_input4;
        input8 = tmp_input8;
    }
    {
        svint32_t inputCopy = input5;
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input7, inputCopy);
        svint32_t tmp_input7 = svmax_s32_z(svptrue_b32(),input7, inputCopy);
        svint32_t input5_val_copy = input5_val;
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, inputCopy), input5_val_copy, input7_val);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, input5_val_copy);

        input5 = tmp_input5;
        input7 = tmp_input7;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t tmp_input3 = svmax_s32_z(svptrue_b32(),input3, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input3_val);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, input_val_copy);

        input = tmp_input;
        input3 = tmp_input3;
    }
    {
        svint32_t inputCopy = input2;
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t input2_val_copy = input2_val;
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, inputCopy), input2_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input2_val_copy);

        input2 = tmp_input2;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input;
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t tmp_input2 = svmax_s32_z(svptrue_b32(),input2, inputCopy);
        svint32_t input_val_copy = input_val;
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, inputCopy), input_val_copy, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input_val_copy);

        input = tmp_input;
        input2 = tmp_input2;
    }
    {
        svint32_t inputCopy = input3;
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t tmp_input4 = svmax_s32_z(svptrue_b32(),input4, inputCopy);
        svint32_t input3_val_copy = input3_val;
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, inputCopy), input3_val_copy, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input3_val_copy);

        input3 = tmp_input3;
        input4 = tmp_input4;
    }
    {
        svint32_t inputCopy = input5;
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input7, inputCopy);
        svint32_t tmp_input7 = svmax_s32_z(svptrue_b32(),input7, inputCopy);
        svint32_t input5_val_copy = input5_val;
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, inputCopy), input5_val_copy, input7_val);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, input5_val_copy);

        input5 = tmp_input5;
        input7 = tmp_input7;
    }
    {
        svint32_t inputCopy = input6;
        svint32_t tmp_input6 = svmin_s32_z(svptrue_b32(),input8, inputCopy);
        svint32_t tmp_input8 = svmax_s32_z(svptrue_b32(),input8, inputCopy);
        svint32_t input6_val_copy = input6_val;
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, inputCopy), input6_val_copy, input8_val);
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, input6_val_copy);

        input6 = tmp_input6;
        input8 = tmp_input8;
    }
    {
        svint32_t inputCopy = input5;
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t tmp_input6 = svmax_s32_z(svptrue_b32(),input6, inputCopy);
        svint32_t input5_val_copy = input5_val;
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, inputCopy), input5_val_copy, input6_val);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, input5_val_copy);

        input5 = tmp_input5;
        input6 = tmp_input6;
    }
    {
        svint32_t inputCopy = input7;
        svint32_t tmp_input7 = svmin_s32_z(svptrue_b32(),input8, inputCopy);
        svint32_t tmp_input8 = svmax_s32_z(svptrue_b32(),input8, inputCopy);
        svint32_t input7_val_copy = input7_val;
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, inputCopy), input7_val_copy, input8_val);
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, input7_val_copy);

        input7 = tmp_input7;
        input8 = tmp_input8;
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
        svint32_t permNeighMax8 = svmax_s32_z(svptrue_b32(),permNeigh8, input8);svint32_t tmp_input = svsel_s32(mask0xFF00,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xFF00,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xFF00,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xFF00,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xFF00,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xFF00,  permNeighMax6, permNeighMin6);
        svint32_t tmp_input7 = svsel_s32(mask0xFF00,  permNeighMax7, permNeighMin7);
        svint32_t tmp_input8 = svsel_s32(mask0xFF00,  permNeighMax8, permNeighMin8);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, svtbl_s32( input7_val, idxNoNeigh));
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, svtbl_s32( input8_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
        input7 = tmp_input7;
        input8 = tmp_input8;
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
        svint32_t tmp_input = svsel_s32(mask0xF0F0,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xF0F0,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xF0F0,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xF0F0,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xF0F0,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xF0F0,  permNeighMax6, permNeighMin6);
        svint32_t tmp_input7 = svsel_s32(mask0xF0F0,  permNeighMax7, permNeighMin7);
        svint32_t tmp_input8 = svsel_s32(mask0xF0F0,  permNeighMax8, permNeighMin8);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, svtbl_s32( input7_val, idxNoNeigh));
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, svtbl_s32( input8_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
        input7 = tmp_input7;
        input8 = tmp_input8;
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
        svint32_t tmp_input = svsel_s32(mask0xCCCC,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xCCCC,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xCCCC,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xCCCC,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xCCCC,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xCCCC,  permNeighMax6, permNeighMin6);
        svint32_t tmp_input7 = svsel_s32(mask0xCCCC,  permNeighMax7, permNeighMin7);
        svint32_t tmp_input8 = svsel_s32(mask0xCCCC,  permNeighMax8, permNeighMin8);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, svtbl_s32( input7_val, idxNoNeigh));
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, svtbl_s32( input8_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
        input7 = tmp_input7;
        input8 = tmp_input8;
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
        svint32_t tmp_input = svsel_s32(mask0xAAAA,  permNeighMax, permNeighMin);
        svint32_t tmp_input2 = svsel_s32(mask0xAAAA,  permNeighMax2, permNeighMin2);
        svint32_t tmp_input3 = svsel_s32(mask0xAAAA,  permNeighMax3, permNeighMin3);
        svint32_t tmp_input4 = svsel_s32(mask0xAAAA,  permNeighMax4, permNeighMin4);
        svint32_t tmp_input5 = svsel_s32(mask0xAAAA,  permNeighMax5, permNeighMin5);
        svint32_t tmp_input6 = svsel_s32(mask0xAAAA,  permNeighMax6, permNeighMin6);
        svint32_t tmp_input7 = svsel_s32(mask0xAAAA,  permNeighMax7, permNeighMin7);
        svint32_t tmp_input8 = svsel_s32(mask0xAAAA,  permNeighMax8, permNeighMin8);

        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, svtbl_s32( input_val, idxNoNeigh));
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, svtbl_s32( input2_val, idxNoNeigh));
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, svtbl_s32( input3_val, idxNoNeigh));
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, svtbl_s32( input4_val, idxNoNeigh));
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, svtbl_s32( input5_val, idxNoNeigh));
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, svtbl_s32( input6_val, idxNoNeigh));
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, svtbl_s32( input7_val, idxNoNeigh));
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, svtbl_s32( input8_val, idxNoNeigh));

        input = tmp_input;
        input2 = tmp_input2;
        input3 = tmp_input3;
        input4 = tmp_input4;
        input5 = tmp_input5;
        input6 = tmp_input6;
        input7 = tmp_input7;
        input8 = tmp_input8;
    }
}


inline void CoreSmallSort9(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9,
                                 svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                                 svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val,
                                 svint32_t& input9_val){
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
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8,
                         input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSort(input9, input9_val);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh9 = svtbl_s32( input9, idxNoNeigh);

        svint32_t tmp_input9 = svmax_s32_z(svptrue_b32(),input8, permNeigh9);
        svint32_t tmp_input8 = svmin_s32_z(svptrue_b32(),input8, permNeigh9);


        svint32_t input9_val_perm = svtbl_s32( input9_val, idxNoNeigh);
        input9_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input9, permNeigh9), input9_val_perm, input8_val);
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, input9_val_perm);

        input9 = tmp_input9;
        input8 = tmp_input8;
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8,
                      input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallEnd1(input9, input9_val);
}


inline void CoreSmallSort9(int* __restrict__ ptr1, int* __restrict__ values ){
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1+0*16);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+1*16);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*16);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*16);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*16);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*16);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*16);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*16);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr1+8*16);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values+0*16);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+1*16);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*16);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*16);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*16);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*16);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*16);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*16);
    svint32_t input8_val = svld1_s32(svptrue_b32(),values+8*16);
    CoreSmallSort9(input0,input1,input2,input3,input4,input5,input6,input7,input8,
        input0_val,input1_val,input2_val,input3_val,input4_val,input5_val,input6_val,input7_val,input8_val);
    svst1_s32(svptrue_b32(),ptr1+0*16, input0);
    svst1_s32(svptrue_b32(),ptr1+1*16, input1);
    svst1_s32(svptrue_b32(),ptr1+2*16, input2);
    svst1_s32(svptrue_b32(),ptr1+3*16, input3);
    svst1_s32(svptrue_b32(),ptr1+4*16, input4);
    svst1_s32(svptrue_b32(),ptr1+5*16, input5);
    svst1_s32(svptrue_b32(),ptr1+6*16, input6);
    svst1_s32(svptrue_b32(),ptr1+7*16, input7);
    svst1_s32(svptrue_b32(),ptr1+8*16, input8);
    svst1_s32(svptrue_b32(),values+0*16, input0_val);
    svst1_s32(svptrue_b32(),values+1*16, input1_val);
    svst1_s32(svptrue_b32(),values+2*16, input2_val);
    svst1_s32(svptrue_b32(),values+3*16, input3_val);
    svst1_s32(svptrue_b32(),values+4*16, input4_val);
    svst1_s32(svptrue_b32(),values+5*16, input5_val);
    svst1_s32(svptrue_b32(),values+6*16, input6_val);
    svst1_s32(svptrue_b32(),values+7*16, input7_val);
    svst1_s32(svptrue_b32(),values+8*16, input8_val);
}


inline void CoreSmallSort10(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10,
                             svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                             svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val,
                             svint32_t& input9_val, svint32_t& input10_val){
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
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8,
                         input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSort2(input9, input10, input9_val, input10_val);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh9 = svtbl_s32( input9, idxNoNeigh);
        svint32_t permNeigh10 = svtbl_s32( input10, idxNoNeigh);

        svint32_t tmp_input9 = svmax_s32_z(svptrue_b32(),input8, permNeigh9);
        svint32_t tmp_input8 = svmin_s32_z(svptrue_b32(),input8, permNeigh9);

        svint32_t tmp_input10 = svmax_s32_z(svptrue_b32(),input7, permNeigh10);
        svint32_t tmp_input7 = svmin_s32_z(svptrue_b32(),input7, permNeigh10);


        svint32_t input9_val_perm = svtbl_s32( input9_val, idxNoNeigh);
        input9_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input9, permNeigh9), input9_val_perm, input8_val);
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, input9_val_perm);

        svint32_t input10_val_perm = svtbl_s32( input10_val, idxNoNeigh);
        input10_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input10, permNeigh10), input10_val_perm, input7_val);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, input10_val_perm);


        input9 = tmp_input9;
        input8 = tmp_input8;

        input10 = tmp_input10;
        input7 = tmp_input7;
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8,
                           input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallEnd2(input9, input10, input9_val, input10_val);
}


inline void CoreSmallSort10(int* __restrict__ ptr1, int* __restrict__ values ){
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1+0*16);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+1*16);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*16);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*16);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*16);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*16);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*16);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*16);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr1+8*16);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr1+9*16);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values+0*16);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+1*16);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*16);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*16);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*16);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*16);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*16);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*16);
    svint32_t input8_val = svld1_s32(svptrue_b32(),values+8*16);
    svint32_t input9_val = svld1_s32(svptrue_b32(),values+9*16);
    CoreSmallSort10(input0,input1,input2,input3,input4,input5,input6,input7,input8,input9,
        input0_val,input1_val,input2_val,input3_val,input4_val,input5_val,input6_val,input7_val,input8_val,input9_val);
    svst1_s32(svptrue_b32(),ptr1+0*16, input0);
    svst1_s32(svptrue_b32(),ptr1+1*16, input1);
    svst1_s32(svptrue_b32(),ptr1+2*16, input2);
    svst1_s32(svptrue_b32(),ptr1+3*16, input3);
    svst1_s32(svptrue_b32(),ptr1+4*16, input4);
    svst1_s32(svptrue_b32(),ptr1+5*16, input5);
    svst1_s32(svptrue_b32(),ptr1+6*16, input6);
    svst1_s32(svptrue_b32(),ptr1+7*16, input7);
    svst1_s32(svptrue_b32(),ptr1+8*16, input8);
    svst1_s32(svptrue_b32(),ptr1+9*16, input9);
    svst1_s32(svptrue_b32(),values+0*16, input0_val);
    svst1_s32(svptrue_b32(),values+1*16, input1_val);
    svst1_s32(svptrue_b32(),values+2*16, input2_val);
    svst1_s32(svptrue_b32(),values+3*16, input3_val);
    svst1_s32(svptrue_b32(),values+4*16, input4_val);
    svst1_s32(svptrue_b32(),values+5*16, input5_val);
    svst1_s32(svptrue_b32(),values+6*16, input6_val);
    svst1_s32(svptrue_b32(),values+7*16, input7_val);
    svst1_s32(svptrue_b32(),values+8*16, input8_val);
    svst1_s32(svptrue_b32(),values+9*16, input9_val);
}



inline void CoreSmallSort11(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11,
                                  svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                                  svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val,
                                  svint32_t& input9_val, svint32_t& input10_val, svint32_t& input11_val){
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
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8,
                    input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSort3(input9, input10, input11,
                    input9_val, input10_val, input11_val);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh9 = svtbl_s32( input9, idxNoNeigh);
        svint32_t permNeigh10 = svtbl_s32( input10, idxNoNeigh);
        svint32_t permNeigh11 = svtbl_s32( input11, idxNoNeigh);

        svint32_t tmp_input9 = svmax_s32_z(svptrue_b32(),input8, permNeigh9);
        svint32_t tmp_input8 = svmin_s32_z(svptrue_b32(),input8, permNeigh9);

        svint32_t tmp_input10 = svmax_s32_z(svptrue_b32(),input7, permNeigh10);
        svint32_t tmp_input7 = svmin_s32_z(svptrue_b32(),input7, permNeigh10);

        svint32_t tmp_input11 = svmax_s32_z(svptrue_b32(),input6, permNeigh11);
        svint32_t tmp_input6 = svmin_s32_z(svptrue_b32(),input6, permNeigh11);


        svint32_t input9_val_perm = svtbl_s32( input9_val, idxNoNeigh);
        input9_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input9, permNeigh9), input9_val_perm, input8_val);
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, input9_val_perm);

        svint32_t input10_val_perm = svtbl_s32( input10_val, idxNoNeigh);
        input10_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input10, permNeigh10), input10_val_perm, input7_val);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, input10_val_perm);

        svint32_t input11_val_perm = svtbl_s32( input11_val, idxNoNeigh);
        input11_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input11, permNeigh11), input11_val_perm, input6_val);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, input11_val_perm);


        input9 = tmp_input9;
        input8 = tmp_input8;

        input10 = tmp_input10;
        input7 = tmp_input7;

        input11 = tmp_input11;
        input6 = tmp_input6;
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8,
                      input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallEnd3(input9, input10, input11,
                      input9_val, input10_val, input11_val);
}

inline void CoreSmallSort11(int* __restrict__ ptr1, int* __restrict__ values ){
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1+0*16);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+1*16);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*16);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*16);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*16);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*16);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*16);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*16);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr1+8*16);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr1+9*16);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr1+10*16);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values+0*16);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+1*16);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*16);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*16);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*16);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*16);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*16);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*16);
    svint32_t input8_val = svld1_s32(svptrue_b32(),values+8*16);
    svint32_t input9_val = svld1_s32(svptrue_b32(),values+9*16);
    svint32_t input10_val = svld1_s32(svptrue_b32(),values+10*16);
    CoreSmallSort11(input0,input1,input2,input3,input4,input5,input6,input7,input8,input9,input10,
        input0_val,input1_val,input2_val,input3_val,input4_val,input5_val,input6_val,input7_val,input8_val,input9_val,input10_val);
    svst1_s32(svptrue_b32(),ptr1+0*16, input0);
    svst1_s32(svptrue_b32(),ptr1+1*16, input1);
    svst1_s32(svptrue_b32(),ptr1+2*16, input2);
    svst1_s32(svptrue_b32(),ptr1+3*16, input3);
    svst1_s32(svptrue_b32(),ptr1+4*16, input4);
    svst1_s32(svptrue_b32(),ptr1+5*16, input5);
    svst1_s32(svptrue_b32(),ptr1+6*16, input6);
    svst1_s32(svptrue_b32(),ptr1+7*16, input7);
    svst1_s32(svptrue_b32(),ptr1+8*16, input8);
    svst1_s32(svptrue_b32(),ptr1+9*16, input9);
    svst1_s32(svptrue_b32(),ptr1+10*16, input10);
    svst1_s32(svptrue_b32(),values+0*16, input0_val);
    svst1_s32(svptrue_b32(),values+1*16, input1_val);
    svst1_s32(svptrue_b32(),values+2*16, input2_val);
    svst1_s32(svptrue_b32(),values+3*16, input3_val);
    svst1_s32(svptrue_b32(),values+4*16, input4_val);
    svst1_s32(svptrue_b32(),values+5*16, input5_val);
    svst1_s32(svptrue_b32(),values+6*16, input6_val);
    svst1_s32(svptrue_b32(),values+7*16, input7_val);
    svst1_s32(svptrue_b32(),values+8*16, input8_val);
    svst1_s32(svptrue_b32(),values+9*16, input9_val);
    svst1_s32(svptrue_b32(),values+10*16, input10_val);
}

inline void CoreSmallSort12(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                                  svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                                  svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val,
                                  svint32_t& input9_val, svint32_t& input10_val, svint32_t& input11_val ,
                                  svint32_t& input12_val){
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
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8,
                    input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSort4(input9, input10, input11, input12,
                    input9_val, input10_val, input11_val, input12_val);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh9 = svtbl_s32( input9, idxNoNeigh);
        svint32_t permNeigh10 = svtbl_s32( input10, idxNoNeigh);
        svint32_t permNeigh11 = svtbl_s32( input11, idxNoNeigh);
        svint32_t permNeigh12 = svtbl_s32( input12, idxNoNeigh);

        svint32_t tmp_input9 = svmax_s32_z(svptrue_b32(),input8, permNeigh9);
        svint32_t tmp_input8 = svmin_s32_z(svptrue_b32(),input8, permNeigh9);

        svint32_t tmp_input10 = svmax_s32_z(svptrue_b32(),input7, permNeigh10);
        svint32_t tmp_input7 = svmin_s32_z(svptrue_b32(),input7, permNeigh10);

        svint32_t tmp_input11 = svmax_s32_z(svptrue_b32(),input6, permNeigh11);
        svint32_t tmp_input6 = svmin_s32_z(svptrue_b32(),input6, permNeigh11);

        svint32_t tmp_input12 = svmax_s32_z(svptrue_b32(),input5, permNeigh12);
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input5, permNeigh12);

        svint32_t input9_val_perm = svtbl_s32( input9_val, idxNoNeigh);
        input9_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input9, permNeigh9), input9_val_perm, input8_val);
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, input9_val_perm);

        svint32_t input10_val_perm = svtbl_s32( input10_val, idxNoNeigh);
        input10_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input10, permNeigh10), input10_val_perm, input7_val);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, input10_val_perm);

        svint32_t input11_val_perm = svtbl_s32( input11_val, idxNoNeigh);
        input11_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input11, permNeigh11), input11_val_perm, input6_val);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, input11_val_perm);

        svint32_t input12_val_perm = svtbl_s32( input12_val, idxNoNeigh);
        input12_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input12, permNeigh12), input12_val_perm, input5_val);
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, input12_val_perm);

        input9 = tmp_input9;
        input8 = tmp_input8;

        input10 = tmp_input10;
        input7 = tmp_input7;

        input11 = tmp_input11;
        input6 = tmp_input6;

        input12 = tmp_input12;
        input5 = tmp_input5;
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8,
                      input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallEnd4(input9, input10, input11, input12,
                      input9_val, input10_val, input11_val, input12_val);
}


inline void CoreSmallSort12(int* __restrict__ ptr1, int* __restrict__ values ){
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1+0*16);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+1*16);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*16);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*16);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*16);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*16);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*16);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*16);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr1+8*16);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr1+9*16);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr1+10*16);
    svint32_t input11 = svld1_s32(svptrue_b32(),ptr1+11*16);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values+0*16);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+1*16);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*16);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*16);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*16);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*16);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*16);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*16);
    svint32_t input8_val = svld1_s32(svptrue_b32(),values+8*16);
    svint32_t input9_val = svld1_s32(svptrue_b32(),values+9*16);
    svint32_t input10_val = svld1_s32(svptrue_b32(),values+10*16);
    svint32_t input11_val = svld1_s32(svptrue_b32(),values+11*16);
    CoreSmallSort12(input0,input1,input2,input3,input4,input5,input6,input7,input8,input9,input10,input11,
        input0_val,input1_val,input2_val,input3_val,input4_val,input5_val,input6_val,input7_val,input8_val,input9_val,input10_val,input11_val);
    svst1_s32(svptrue_b32(),ptr1+0*16, input0);
    svst1_s32(svptrue_b32(),ptr1+1*16, input1);
    svst1_s32(svptrue_b32(),ptr1+2*16, input2);
    svst1_s32(svptrue_b32(),ptr1+3*16, input3);
    svst1_s32(svptrue_b32(),ptr1+4*16, input4);
    svst1_s32(svptrue_b32(),ptr1+5*16, input5);
    svst1_s32(svptrue_b32(),ptr1+6*16, input6);
    svst1_s32(svptrue_b32(),ptr1+7*16, input7);
    svst1_s32(svptrue_b32(),ptr1+8*16, input8);
    svst1_s32(svptrue_b32(),ptr1+9*16, input9);
    svst1_s32(svptrue_b32(),ptr1+10*16, input10);
    svst1_s32(svptrue_b32(),ptr1+11*16, input11);
    svst1_s32(svptrue_b32(),values+0*16, input0_val);
    svst1_s32(svptrue_b32(),values+1*16, input1_val);
    svst1_s32(svptrue_b32(),values+2*16, input2_val);
    svst1_s32(svptrue_b32(),values+3*16, input3_val);
    svst1_s32(svptrue_b32(),values+4*16, input4_val);
    svst1_s32(svptrue_b32(),values+5*16, input5_val);
    svst1_s32(svptrue_b32(),values+6*16, input6_val);
    svst1_s32(svptrue_b32(),values+7*16, input7_val);
    svst1_s32(svptrue_b32(),values+8*16, input8_val);
    svst1_s32(svptrue_b32(),values+9*16, input9_val);
    svst1_s32(svptrue_b32(),values+10*16, input10_val);
    svst1_s32(svptrue_b32(),values+11*16, input11_val);
}



inline void CoreSmallSort13(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                            svint32_t& input13,
                                  svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                                  svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val,
                                  svint32_t& input9_val, svint32_t& input10_val, svint32_t& input11_val ,
                                  svint32_t& input12_val, svint32_t& input13_val){
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
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8,
                    input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSort5(input9, input10, input11, input12, input13,
                    input9_val, input10_val, input11_val, input12_val, input13_val);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh9 = svtbl_s32( input9, idxNoNeigh);
        svint32_t permNeigh10 = svtbl_s32( input10, idxNoNeigh);
        svint32_t permNeigh11 = svtbl_s32( input11, idxNoNeigh);
        svint32_t permNeigh12 = svtbl_s32( input12, idxNoNeigh);
        svint32_t permNeigh13 = svtbl_s32( input13, idxNoNeigh);

        svint32_t tmp_input9 = svmax_s32_z(svptrue_b32(),input8, permNeigh9);
        svint32_t tmp_input8 = svmin_s32_z(svptrue_b32(),input8, permNeigh9);

        svint32_t tmp_input10 = svmax_s32_z(svptrue_b32(),input7, permNeigh10);
        svint32_t tmp_input7 = svmin_s32_z(svptrue_b32(),input7, permNeigh10);

        svint32_t tmp_input11 = svmax_s32_z(svptrue_b32(),input6, permNeigh11);
        svint32_t tmp_input6 = svmin_s32_z(svptrue_b32(),input6, permNeigh11);

        svint32_t tmp_input12 = svmax_s32_z(svptrue_b32(),input5, permNeigh12);
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input5, permNeigh12);

        svint32_t tmp_input13 = svmax_s32_z(svptrue_b32(),input4, permNeigh13);
        svint32_t tmp_input4 = svmin_s32_z(svptrue_b32(),input4, permNeigh13);

        svint32_t input9_val_perm = svtbl_s32( input9_val, idxNoNeigh);
        input9_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input9, permNeigh9), input9_val_perm, input8_val);
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, input9_val_perm);

        svint32_t input10_val_perm = svtbl_s32( input10_val, idxNoNeigh);
        input10_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input10, permNeigh10), input10_val_perm, input7_val);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, input10_val_perm);

        svint32_t input11_val_perm = svtbl_s32( input11_val, idxNoNeigh);
        input11_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input11, permNeigh11), input11_val_perm, input6_val);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, input11_val_perm);

        svint32_t input12_val_perm = svtbl_s32( input12_val, idxNoNeigh);
        input12_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input12, permNeigh12), input12_val_perm, input5_val);
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, input12_val_perm);

        svint32_t input13_val_perm = svtbl_s32( input13_val, idxNoNeigh);
        input13_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input13, permNeigh13), input13_val_perm, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input13_val_perm);

        input9 = tmp_input9;
        input8 = tmp_input8;

        input10 = tmp_input10;
        input7 = tmp_input7;

        input11 = tmp_input11;
        input6 = tmp_input6;

        input12 = tmp_input12;
        input5 = tmp_input5;

        input13 = tmp_input13;
        input4 = tmp_input4;
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8,
                      input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallEnd5(input9, input10, input11, input12, input13,
                      input9_val, input10_val, input11_val, input12_val, input13_val);
}


inline void CoreSmallSort13(int* __restrict__ ptr1, int* __restrict__ values ){
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1+0*16);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+1*16);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*16);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*16);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*16);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*16);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*16);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*16);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr1+8*16);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr1+9*16);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr1+10*16);
    svint32_t input11 = svld1_s32(svptrue_b32(),ptr1+11*16);
    svint32_t input12 = svld1_s32(svptrue_b32(),ptr1+12*16);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values+0*16);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+1*16);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*16);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*16);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*16);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*16);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*16);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*16);
    svint32_t input8_val = svld1_s32(svptrue_b32(),values+8*16);
    svint32_t input9_val = svld1_s32(svptrue_b32(),values+9*16);
    svint32_t input10_val = svld1_s32(svptrue_b32(),values+10*16);
    svint32_t input11_val = svld1_s32(svptrue_b32(),values+11*16);
    svint32_t input12_val = svld1_s32(svptrue_b32(),values+12*16);
    CoreSmallSort13(input0,input1,input2,input3,input4,input5,input6,input7,input8,input9,input10,input11,input12,
        input0_val,input1_val,input2_val,input3_val,input4_val,input5_val,input6_val,input7_val,input8_val,input9_val,input10_val,input11_val,input12_val);
    svst1_s32(svptrue_b32(),ptr1+0*16, input0);
    svst1_s32(svptrue_b32(),ptr1+1*16, input1);
    svst1_s32(svptrue_b32(),ptr1+2*16, input2);
    svst1_s32(svptrue_b32(),ptr1+3*16, input3);
    svst1_s32(svptrue_b32(),ptr1+4*16, input4);
    svst1_s32(svptrue_b32(),ptr1+5*16, input5);
    svst1_s32(svptrue_b32(),ptr1+6*16, input6);
    svst1_s32(svptrue_b32(),ptr1+7*16, input7);
    svst1_s32(svptrue_b32(),ptr1+8*16, input8);
    svst1_s32(svptrue_b32(),ptr1+9*16, input9);
    svst1_s32(svptrue_b32(),ptr1+10*16, input10);
    svst1_s32(svptrue_b32(),ptr1+11*16, input11);
    svst1_s32(svptrue_b32(),ptr1+12*16, input12);
    svst1_s32(svptrue_b32(),values+0*16, input0_val);
    svst1_s32(svptrue_b32(),values+1*16, input1_val);
    svst1_s32(svptrue_b32(),values+2*16, input2_val);
    svst1_s32(svptrue_b32(),values+3*16, input3_val);
    svst1_s32(svptrue_b32(),values+4*16, input4_val);
    svst1_s32(svptrue_b32(),values+5*16, input5_val);
    svst1_s32(svptrue_b32(),values+6*16, input6_val);
    svst1_s32(svptrue_b32(),values+7*16, input7_val);
    svst1_s32(svptrue_b32(),values+8*16, input8_val);
    svst1_s32(svptrue_b32(),values+9*16, input9_val);
    svst1_s32(svptrue_b32(),values+10*16, input10_val);
    svst1_s32(svptrue_b32(),values+11*16, input11_val);
    svst1_s32(svptrue_b32(),values+12*16, input12_val);
}



inline void CoreSmallSort14(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                            svint32_t& input13, svint32_t& input14,
                                  svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                                  svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val,
                                  svint32_t& input9_val, svint32_t& input10_val, svint32_t& input11_val ,
                                  svint32_t& input12_val, svint32_t& input13_val, svint32_t& input14_val){
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
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8,
                    input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSort6(input9, input10, input11, input12, input13, input14,
                    input9_val, input10_val, input11_val, input12_val, input13_val, input14_val);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh9 = svtbl_s32( input9, idxNoNeigh);
        svint32_t permNeigh10 = svtbl_s32( input10, idxNoNeigh);
        svint32_t permNeigh11 = svtbl_s32( input11, idxNoNeigh);
        svint32_t permNeigh12 = svtbl_s32( input12, idxNoNeigh);
        svint32_t permNeigh13 = svtbl_s32( input13, idxNoNeigh);
        svint32_t permNeigh14 = svtbl_s32( input14, idxNoNeigh);

        svint32_t tmp_input9 = svmax_s32_z(svptrue_b32(),input8, permNeigh9);
        svint32_t tmp_input8 = svmin_s32_z(svptrue_b32(),input8, permNeigh9);

        svint32_t tmp_input10 = svmax_s32_z(svptrue_b32(),input7, permNeigh10);
        svint32_t tmp_input7 = svmin_s32_z(svptrue_b32(),input7, permNeigh10);

        svint32_t tmp_input11 = svmax_s32_z(svptrue_b32(),input6, permNeigh11);
        svint32_t tmp_input6 = svmin_s32_z(svptrue_b32(),input6, permNeigh11);

        svint32_t tmp_input12 = svmax_s32_z(svptrue_b32(),input5, permNeigh12);
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input5, permNeigh12);

        svint32_t tmp_input13 = svmax_s32_z(svptrue_b32(),input4, permNeigh13);
        svint32_t tmp_input4 = svmin_s32_z(svptrue_b32(),input4, permNeigh13);

        svint32_t tmp_input14 = svmax_s32_z(svptrue_b32(),input3, permNeigh14);
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input3, permNeigh14);

        svint32_t input9_val_perm = svtbl_s32( input9_val, idxNoNeigh);
        input9_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input9, permNeigh9), input9_val_perm, input8_val);
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, input9_val_perm);

        svint32_t input10_val_perm = svtbl_s32( input10_val, idxNoNeigh);
        input10_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input10, permNeigh10), input10_val_perm, input7_val);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, input10_val_perm);

        svint32_t input11_val_perm = svtbl_s32( input11_val, idxNoNeigh);
        input11_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input11, permNeigh11), input11_val_perm, input6_val);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, input11_val_perm);

        svint32_t input12_val_perm = svtbl_s32( input12_val, idxNoNeigh);
        input12_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input12, permNeigh12), input12_val_perm, input5_val);
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, input12_val_perm);

        svint32_t input13_val_perm = svtbl_s32( input13_val, idxNoNeigh);
        input13_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input13, permNeigh13), input13_val_perm, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input13_val_perm);

        svint32_t input14_val_perm = svtbl_s32( input14_val, idxNoNeigh);
        input14_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input14, permNeigh14), input14_val_perm, input3_val);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, input14_val_perm);

        input9 = tmp_input9;
        input8 = tmp_input8;

        input10 = tmp_input10;
        input7 = tmp_input7;

        input11 = tmp_input11;
        input6 = tmp_input6;

        input12 = tmp_input12;
        input5 = tmp_input5;

        input13 = tmp_input13;
        input4 = tmp_input4;

        input14 = tmp_input14;
        input3 = tmp_input3;
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8,
                      input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallEnd6(input9, input10, input11, input12, input13, input14,
                      input9_val, input10_val, input11_val, input12_val, input13_val, input14_val);
}


inline void CoreSmallSort14(int* __restrict__ ptr1, int* __restrict__ values ){
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1+0*16);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+1*16);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*16);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*16);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*16);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*16);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*16);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*16);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr1+8*16);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr1+9*16);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr1+10*16);
    svint32_t input11 = svld1_s32(svptrue_b32(),ptr1+11*16);
    svint32_t input12 = svld1_s32(svptrue_b32(),ptr1+12*16);
    svint32_t input13 = svld1_s32(svptrue_b32(),ptr1+13*16);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values+0*16);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+1*16);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*16);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*16);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*16);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*16);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*16);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*16);
    svint32_t input8_val = svld1_s32(svptrue_b32(),values+8*16);
    svint32_t input9_val = svld1_s32(svptrue_b32(),values+9*16);
    svint32_t input10_val = svld1_s32(svptrue_b32(),values+10*16);
    svint32_t input11_val = svld1_s32(svptrue_b32(),values+11*16);
    svint32_t input12_val = svld1_s32(svptrue_b32(),values+12*16);
    svint32_t input13_val = svld1_s32(svptrue_b32(),values+13*16);
    CoreSmallSort14(input0,input1,input2,input3,input4,input5,input6,input7,input8,input9,input10,input11,input12,input13,
        input0_val,input1_val,input2_val,input3_val,input4_val,input5_val,input6_val,input7_val,input8_val,input9_val,input10_val,input11_val,input12_val,input13_val);
    svst1_s32(svptrue_b32(),ptr1+0*16, input0);
    svst1_s32(svptrue_b32(),ptr1+1*16, input1);
    svst1_s32(svptrue_b32(),ptr1+2*16, input2);
    svst1_s32(svptrue_b32(),ptr1+3*16, input3);
    svst1_s32(svptrue_b32(),ptr1+4*16, input4);
    svst1_s32(svptrue_b32(),ptr1+5*16, input5);
    svst1_s32(svptrue_b32(),ptr1+6*16, input6);
    svst1_s32(svptrue_b32(),ptr1+7*16, input7);
    svst1_s32(svptrue_b32(),ptr1+8*16, input8);
    svst1_s32(svptrue_b32(),ptr1+9*16, input9);
    svst1_s32(svptrue_b32(),ptr1+10*16, input10);
    svst1_s32(svptrue_b32(),ptr1+11*16, input11);
    svst1_s32(svptrue_b32(),ptr1+12*16, input12);
    svst1_s32(svptrue_b32(),ptr1+13*16, input13);
    svst1_s32(svptrue_b32(),values+0*16, input0_val);
    svst1_s32(svptrue_b32(),values+1*16, input1_val);
    svst1_s32(svptrue_b32(),values+2*16, input2_val);
    svst1_s32(svptrue_b32(),values+3*16, input3_val);
    svst1_s32(svptrue_b32(),values+4*16, input4_val);
    svst1_s32(svptrue_b32(),values+5*16, input5_val);
    svst1_s32(svptrue_b32(),values+6*16, input6_val);
    svst1_s32(svptrue_b32(),values+7*16, input7_val);
    svst1_s32(svptrue_b32(),values+8*16, input8_val);
    svst1_s32(svptrue_b32(),values+9*16, input9_val);
    svst1_s32(svptrue_b32(),values+10*16, input10_val);
    svst1_s32(svptrue_b32(),values+11*16, input11_val);
    svst1_s32(svptrue_b32(),values+12*16, input12_val);
    svst1_s32(svptrue_b32(),values+13*16, input13_val);
}


inline void CoreSmallSort15(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                            svint32_t& input13, svint32_t& input14, svint32_t& input15,
                                  svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                                  svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val,
                                  svint32_t& input9_val, svint32_t& input10_val, svint32_t& input11_val ,
                                  svint32_t& input12_val, svint32_t& input13_val, svint32_t& input14_val,
                                  svint32_t& input15_val){
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
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8,
                    input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSort7(input9, input10, input11, input12, input13, input14, input15,
                    input9_val, input10_val, input11_val, input12_val, input13_val, input14_val, input15_val);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh9 = svtbl_s32( input9, idxNoNeigh);
        svint32_t permNeigh10 = svtbl_s32( input10, idxNoNeigh);
        svint32_t permNeigh11 = svtbl_s32( input11, idxNoNeigh);
        svint32_t permNeigh12 = svtbl_s32( input12, idxNoNeigh);
        svint32_t permNeigh13 = svtbl_s32( input13, idxNoNeigh);
        svint32_t permNeigh14 = svtbl_s32( input14, idxNoNeigh);
        svint32_t permNeigh15 = svtbl_s32( input15, idxNoNeigh);

        svint32_t tmp_input9 = svmax_s32_z(svptrue_b32(),input8, permNeigh9);
        svint32_t tmp_input8 = svmin_s32_z(svptrue_b32(),input8, permNeigh9);

        svint32_t tmp_input10 = svmax_s32_z(svptrue_b32(),input7, permNeigh10);
        svint32_t tmp_input7 = svmin_s32_z(svptrue_b32(),input7, permNeigh10);

        svint32_t tmp_input11 = svmax_s32_z(svptrue_b32(),input6, permNeigh11);
        svint32_t tmp_input6 = svmin_s32_z(svptrue_b32(),input6, permNeigh11);

        svint32_t tmp_input12 = svmax_s32_z(svptrue_b32(),input5, permNeigh12);
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input5, permNeigh12);

        svint32_t tmp_input13 = svmax_s32_z(svptrue_b32(),input4, permNeigh13);
        svint32_t tmp_input4 = svmin_s32_z(svptrue_b32(),input4, permNeigh13);

        svint32_t tmp_input14 = svmax_s32_z(svptrue_b32(),input3, permNeigh14);
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input3, permNeigh14);

        svint32_t tmp_input15 = svmax_s32_z(svptrue_b32(),input2, permNeigh15);
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input2, permNeigh15);

        svint32_t input9_val_perm = svtbl_s32( input9_val, idxNoNeigh);
        input9_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input9, permNeigh9), input9_val_perm, input8_val);
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, input9_val_perm);

        svint32_t input10_val_perm = svtbl_s32( input10_val, idxNoNeigh);
        input10_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input10, permNeigh10), input10_val_perm, input7_val);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, input10_val_perm);

        svint32_t input11_val_perm = svtbl_s32( input11_val, idxNoNeigh);
        input11_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input11, permNeigh11), input11_val_perm, input6_val);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, input11_val_perm);

        svint32_t input12_val_perm = svtbl_s32( input12_val, idxNoNeigh);
        input12_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input12, permNeigh12), input12_val_perm, input5_val);
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, input12_val_perm);

        svint32_t input13_val_perm = svtbl_s32( input13_val, idxNoNeigh);
        input13_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input13, permNeigh13), input13_val_perm, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input13_val_perm);

        svint32_t input14_val_perm = svtbl_s32( input14_val, idxNoNeigh);
        input14_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input14, permNeigh14), input14_val_perm, input3_val);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, input14_val_perm);

        svint32_t input15_val_perm = svtbl_s32( input15_val, idxNoNeigh);
        input15_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input15, permNeigh15), input15_val_perm, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input15_val_perm);

        input9 = tmp_input9;
        input8 = tmp_input8;

        input10 = tmp_input10;
        input7 = tmp_input7;

        input11 = tmp_input11;
        input6 = tmp_input6;

        input12 = tmp_input12;
        input5 = tmp_input5;

        input13 = tmp_input13;
        input4 = tmp_input4;

        input14 = tmp_input14;
        input3 = tmp_input3;

        input15 = tmp_input15;
        input2 = tmp_input2;
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8,
                      input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallEnd7(input9, input10, input11, input12, input13, input14, input15,
                      input9_val, input10_val, input11_val, input12_val, input13_val, input14_val, input15_val);
}


inline void CoreSmallSort15(int* __restrict__ ptr1, int* __restrict__ values ){
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1+0*16);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+1*16);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*16);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*16);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*16);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*16);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*16);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*16);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr1+8*16);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr1+9*16);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr1+10*16);
    svint32_t input11 = svld1_s32(svptrue_b32(),ptr1+11*16);
    svint32_t input12 = svld1_s32(svptrue_b32(),ptr1+12*16);
    svint32_t input13 = svld1_s32(svptrue_b32(),ptr1+13*16);
    svint32_t input14 = svld1_s32(svptrue_b32(),ptr1+14*16);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values+0*16);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+1*16);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*16);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*16);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*16);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*16);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*16);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*16);
    svint32_t input8_val = svld1_s32(svptrue_b32(),values+8*16);
    svint32_t input9_val = svld1_s32(svptrue_b32(),values+9*16);
    svint32_t input10_val = svld1_s32(svptrue_b32(),values+10*16);
    svint32_t input11_val = svld1_s32(svptrue_b32(),values+11*16);
    svint32_t input12_val = svld1_s32(svptrue_b32(),values+12*16);
    svint32_t input13_val = svld1_s32(svptrue_b32(),values+13*16);
    svint32_t input14_val = svld1_s32(svptrue_b32(),values+14*16);
    CoreSmallSort15(input0,input1,input2,input3,input4,input5,input6,input7,input8,input9,input10,input11,input12,input13,input14,
        input0_val,input1_val,input2_val,input3_val,input4_val,input5_val,input6_val,input7_val,input8_val,input9_val,input10_val,input11_val,input12_val,input13_val,input14_val);
    svst1_s32(svptrue_b32(),ptr1+0*16, input0);
    svst1_s32(svptrue_b32(),ptr1+1*16, input1);
    svst1_s32(svptrue_b32(),ptr1+2*16, input2);
    svst1_s32(svptrue_b32(),ptr1+3*16, input3);
    svst1_s32(svptrue_b32(),ptr1+4*16, input4);
    svst1_s32(svptrue_b32(),ptr1+5*16, input5);
    svst1_s32(svptrue_b32(),ptr1+6*16, input6);
    svst1_s32(svptrue_b32(),ptr1+7*16, input7);
    svst1_s32(svptrue_b32(),ptr1+8*16, input8);
    svst1_s32(svptrue_b32(),ptr1+9*16, input9);
    svst1_s32(svptrue_b32(),ptr1+10*16, input10);
    svst1_s32(svptrue_b32(),ptr1+11*16, input11);
    svst1_s32(svptrue_b32(),ptr1+12*16, input12);
    svst1_s32(svptrue_b32(),ptr1+13*16, input13);
    svst1_s32(svptrue_b32(),ptr1+14*16, input14);
    svst1_s32(svptrue_b32(),values+0*16, input0_val);
    svst1_s32(svptrue_b32(),values+1*16, input1_val);
    svst1_s32(svptrue_b32(),values+2*16, input2_val);
    svst1_s32(svptrue_b32(),values+3*16, input3_val);
    svst1_s32(svptrue_b32(),values+4*16, input4_val);
    svst1_s32(svptrue_b32(),values+5*16, input5_val);
    svst1_s32(svptrue_b32(),values+6*16, input6_val);
    svst1_s32(svptrue_b32(),values+7*16, input7_val);
    svst1_s32(svptrue_b32(),values+8*16, input8_val);
    svst1_s32(svptrue_b32(),values+9*16, input9_val);
    svst1_s32(svptrue_b32(),values+10*16, input10_val);
    svst1_s32(svptrue_b32(),values+11*16, input11_val);
    svst1_s32(svptrue_b32(),values+12*16, input12_val);
    svst1_s32(svptrue_b32(),values+13*16, input13_val);
    svst1_s32(svptrue_b32(),values+14*16, input14_val);
}



inline void CoreSmallSort16(svint32_t& input, svint32_t& input2, svint32_t& input3, svint32_t& input4,
                            svint32_t& input5, svint32_t& input6, svint32_t& input7, svint32_t& input8,
                            svint32_t& input9, svint32_t& input10, svint32_t& input11, svint32_t& input12,
                            svint32_t& input13, svint32_t& input14, svint32_t& input15, svint32_t& input16,
                                  svint32_t& input_val, svint32_t& input2_val, svint32_t& input3_val, svint32_t& input4_val,
                                  svint32_t& input5_val, svint32_t& input6_val, svint32_t& input7_val, svint32_t& input8_val,
                                  svint32_t& input9_val, svint32_t& input10_val, svint32_t& input11_val ,
                                  svint32_t& input12_val, svint32_t& input13_val, svint32_t& input14_val,
                                  svint32_t& input15_val,svint32_t& input16_val){
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
    CoreSmallSort8(input, input2, input3, input4, input5, input6, input7, input8,
                    input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallSort8(input9, input10, input11, input12, input13, input14, input15, input16,
                    input9_val, input10_val, input11_val, input12_val, input13_val, input14_val, input15_val, input16_val);
    {
        svuint32_t idxNoNeigh = svld1_u32(svptrue_b32(), array0123456789101112131415);
        svint32_t permNeigh9 = svtbl_s32( input9, idxNoNeigh);
        svint32_t permNeigh10 = svtbl_s32( input10, idxNoNeigh);
        svint32_t permNeigh11 = svtbl_s32( input11, idxNoNeigh);
        svint32_t permNeigh12 = svtbl_s32( input12, idxNoNeigh);
        svint32_t permNeigh13 = svtbl_s32( input13, idxNoNeigh);
        svint32_t permNeigh14 = svtbl_s32( input14, idxNoNeigh);
        svint32_t permNeigh15 = svtbl_s32( input15, idxNoNeigh);
        svint32_t permNeigh16 = svtbl_s32( input16, idxNoNeigh);

        svint32_t tmp_input9 = svmax_s32_z(svptrue_b32(),input8, permNeigh9);
        svint32_t tmp_input8 = svmin_s32_z(svptrue_b32(),input8, permNeigh9);

        svint32_t tmp_input10 = svmax_s32_z(svptrue_b32(),input7, permNeigh10);
        svint32_t tmp_input7 = svmin_s32_z(svptrue_b32(),input7, permNeigh10);

        svint32_t tmp_input11 = svmax_s32_z(svptrue_b32(),input6, permNeigh11);
        svint32_t tmp_input6 = svmin_s32_z(svptrue_b32(),input6, permNeigh11);

        svint32_t tmp_input12 = svmax_s32_z(svptrue_b32(),input5, permNeigh12);
        svint32_t tmp_input5 = svmin_s32_z(svptrue_b32(),input5, permNeigh12);

        svint32_t tmp_input13 = svmax_s32_z(svptrue_b32(),input4, permNeigh13);
        svint32_t tmp_input4 = svmin_s32_z(svptrue_b32(),input4, permNeigh13);

        svint32_t tmp_input14 = svmax_s32_z(svptrue_b32(),input3, permNeigh14);
        svint32_t tmp_input3 = svmin_s32_z(svptrue_b32(),input3, permNeigh14);

        svint32_t tmp_input15 = svmax_s32_z(svptrue_b32(),input2, permNeigh15);
        svint32_t tmp_input2 = svmin_s32_z(svptrue_b32(),input2, permNeigh15);

        svint32_t tmp_input16 = svmax_s32_z(svptrue_b32(),input, permNeigh16);
        svint32_t tmp_input = svmin_s32_z(svptrue_b32(),input, permNeigh16);


        svint32_t input9_val_perm = svtbl_s32( input9_val, idxNoNeigh);
        input9_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input9, permNeigh9), input9_val_perm, input8_val);
        input8_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input8, input8), input8_val, input9_val_perm);

        svint32_t input10_val_perm = svtbl_s32( input10_val, idxNoNeigh);
        input10_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input10, permNeigh10), input10_val_perm, input7_val);
        input7_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input7, input7), input7_val, input10_val_perm);

        svint32_t input11_val_perm = svtbl_s32( input11_val, idxNoNeigh);
        input11_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input11, permNeigh11), input11_val_perm, input6_val);
        input6_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input6, input6), input6_val, input11_val_perm);

        svint32_t input12_val_perm = svtbl_s32( input12_val, idxNoNeigh);
        input12_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input12, permNeigh12), input12_val_perm, input5_val);
        input5_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input5, input5), input5_val, input12_val_perm);

        svint32_t input13_val_perm = svtbl_s32( input13_val, idxNoNeigh);
        input13_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input13, permNeigh13), input13_val_perm, input4_val);
        input4_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input4, input4), input4_val, input13_val_perm);

        svint32_t input14_val_perm = svtbl_s32( input14_val, idxNoNeigh);
        input14_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input14, permNeigh14), input14_val_perm, input3_val);
        input3_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input3, input3), input3_val, input14_val_perm);

        svint32_t input15_val_perm = svtbl_s32( input15_val, idxNoNeigh);
        input15_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input15, permNeigh15), input15_val_perm, input2_val);
        input2_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input2, input2), input2_val, input15_val_perm);

        svint32_t input16_val_perm = svtbl_s32( input16_val, idxNoNeigh);
        input16_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input16, permNeigh16), input16_val_perm, input_val);
        input_val = svsel_s32(svcmpeq_s32(svptrue_b32(), tmp_input, input), input_val, input16_val_perm);

        input9 = tmp_input9;
        input8 = tmp_input8;

        input10 = tmp_input10;
        input7 = tmp_input7;

        input11 = tmp_input11;
        input6 = tmp_input6;

        input12 = tmp_input12;
        input5 = tmp_input5;

        input13 = tmp_input13;
        input4 = tmp_input4;

        input14 = tmp_input14;
        input3 = tmp_input3;

        input15 = tmp_input15;
        input2 = tmp_input2;

        input16 = tmp_input16;
        input = tmp_input;
    }
    CoreSmallEnd8(input, input2, input3, input4, input5, input6, input7, input8,
                      input_val, input2_val, input3_val, input4_val, input5_val, input6_val, input7_val, input8_val);
    CoreSmallEnd8(input9, input10, input11, input12, input13, input14, input15, input16,
                      input9_val, input10_val, input11_val, input12_val, input13_val, input14_val, input15_val, input16_val);
}


inline void CoreSmallSort16(int* __restrict__ ptr1, int* __restrict__ values ){
    svint32_t input0 = svld1_s32(svptrue_b32(),ptr1+0*16);
    svint32_t input1 = svld1_s32(svptrue_b32(),ptr1+1*16);
    svint32_t input2 = svld1_s32(svptrue_b32(),ptr1+2*16);
    svint32_t input3 = svld1_s32(svptrue_b32(),ptr1+3*16);
    svint32_t input4 = svld1_s32(svptrue_b32(),ptr1+4*16);
    svint32_t input5 = svld1_s32(svptrue_b32(),ptr1+5*16);
    svint32_t input6 = svld1_s32(svptrue_b32(),ptr1+6*16);
    svint32_t input7 = svld1_s32(svptrue_b32(),ptr1+7*16);
    svint32_t input8 = svld1_s32(svptrue_b32(),ptr1+8*16);
    svint32_t input9 = svld1_s32(svptrue_b32(),ptr1+9*16);
    svint32_t input10 = svld1_s32(svptrue_b32(),ptr1+10*16);
    svint32_t input11 = svld1_s32(svptrue_b32(),ptr1+11*16);
    svint32_t input12 = svld1_s32(svptrue_b32(),ptr1+12*16);
    svint32_t input13 = svld1_s32(svptrue_b32(),ptr1+13*16);
    svint32_t input14 = svld1_s32(svptrue_b32(),ptr1+14*16);
    svint32_t input15 = svld1_s32(svptrue_b32(),ptr1+15*16);
    svint32_t input0_val = svld1_s32(svptrue_b32(),values+0*16);
    svint32_t input1_val = svld1_s32(svptrue_b32(),values+1*16);
    svint32_t input2_val = svld1_s32(svptrue_b32(),values+2*16);
    svint32_t input3_val = svld1_s32(svptrue_b32(),values+3*16);
    svint32_t input4_val = svld1_s32(svptrue_b32(),values+4*16);
    svint32_t input5_val = svld1_s32(svptrue_b32(),values+5*16);
    svint32_t input6_val = svld1_s32(svptrue_b32(),values+6*16);
    svint32_t input7_val = svld1_s32(svptrue_b32(),values+7*16);
    svint32_t input8_val = svld1_s32(svptrue_b32(),values+8*16);
    svint32_t input9_val = svld1_s32(svptrue_b32(),values+9*16);
    svint32_t input10_val = svld1_s32(svptrue_b32(),values+10*16);
    svint32_t input11_val = svld1_s32(svptrue_b32(),values+11*16);
    svint32_t input12_val = svld1_s32(svptrue_b32(),values+12*16);
    svint32_t input13_val = svld1_s32(svptrue_b32(),values+13*16);
    svint32_t input14_val = svld1_s32(svptrue_b32(),values+14*16);
    svint32_t input15_val = svld1_s32(svptrue_b32(),values+15*16);
    CoreSmallSort16(input0,input1,input2,input3,input4,input5,input6,input7,input8,input9,input10,input11,input12,input13,input14,input15,
        input0_val,input1_val,input2_val,input3_val,input4_val,input5_val,input6_val,input7_val,input8_val,input9_val,input10_val,input11_val,input12_val,input13_val,input14_val,input15_val);
    svst1_s32(svptrue_b32(),ptr1+0*16, input0);
    svst1_s32(svptrue_b32(),ptr1+1*16, input1);
    svst1_s32(svptrue_b32(),ptr1+2*16, input2);
    svst1_s32(svptrue_b32(),ptr1+3*16, input3);
    svst1_s32(svptrue_b32(),ptr1+4*16, input4);
    svst1_s32(svptrue_b32(),ptr1+5*16, input5);
    svst1_s32(svptrue_b32(),ptr1+6*16, input6);
    svst1_s32(svptrue_b32(),ptr1+7*16, input7);
    svst1_s32(svptrue_b32(),ptr1+8*16, input8);
    svst1_s32(svptrue_b32(),ptr1+9*16, input9);
    svst1_s32(svptrue_b32(),ptr1+10*16, input10);
    svst1_s32(svptrue_b32(),ptr1+11*16, input11);
    svst1_s32(svptrue_b32(),ptr1+12*16, input12);
    svst1_s32(svptrue_b32(),ptr1+13*16, input13);
    svst1_s32(svptrue_b32(),ptr1+14*16, input14);
    svst1_s32(svptrue_b32(),ptr1+15*16, input15);
    svst1_s32(svptrue_b32(),values+0*16, input0_val);
    svst1_s32(svptrue_b32(),values+1*16, input1_val);
    svst1_s32(svptrue_b32(),values+2*16, input2_val);
    svst1_s32(svptrue_b32(),values+3*16, input3_val);
    svst1_s32(svptrue_b32(),values+4*16, input4_val);
    svst1_s32(svptrue_b32(),values+5*16, input5_val);
    svst1_s32(svptrue_b32(),values+6*16, input6_val);
    svst1_s32(svptrue_b32(),values+7*16, input7_val);
    svst1_s32(svptrue_b32(),values+8*16, input8_val);
    svst1_s32(svptrue_b32(),values+9*16, input9_val);
    svst1_s32(svptrue_b32(),values+10*16, input10_val);
    svst1_s32(svptrue_b32(),values+11*16, input11_val);
    svst1_s32(svptrue_b32(),values+12*16, input12_val);
    svst1_s32(svptrue_b32(),values+13*16, input13_val);
    svst1_s32(svptrue_b32(),values+14*16, input14_val);
    svst1_s32(svptrue_b32(),values+15*16, input15_val);
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
static inline IndexType CoreSortGetPivot(const std::pair<SortType,SortType> array[], const IndexType left, const IndexType right){
    const IndexType middle = ((right-left)/2) + left;
    if(array[left].first <= array[middle].first && array[middle].first <= array[right].first){
        return middle;
    }
    else if(array[middle].first <= array[left].first && array[left].first <= array[right].first){
        return left;
    }
    else return right;
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
        if( deep ){
            // default(none) has been removed for clang compatibility
            if(part+1 < right){
#pragma omp task default(shared) firstprivate(array, part, right, deep)
                CoreSortTask<SortType,IndexType>(array, part+1,right, deep - 1);
            }
            // not task needed, let the current thread compute it
            if(part && left < part-1)  CoreSortTask<SortType,IndexType>(array, left,part - 1, deep - 1);
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
        if( deep ){
            // default(none) has been removed for clang compatibility
            if(part+1 < right){
#pragma omp task default(shared) firstprivate(array, values, part, right, deep)
                CoreSortTask<SortType,IndexType>(array,values, part+1,right, deep - 1);
            }
            // not task needed, let the current thread compute it
            if(part && left < part-1)  CoreSortTask<SortType,IndexType>(array,values, left,part - 1, deep - 1);
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
