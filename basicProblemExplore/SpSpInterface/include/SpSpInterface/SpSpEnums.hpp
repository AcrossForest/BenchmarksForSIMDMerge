#pragma once
namespace SPSPEnum{
    enum class EnumEndType
    {
        INCOMPLETE = 0,
        FINISHED = 1
    };

    enum class EnumInxMatMethod
    {
        AND = 0,
        OR = 1,
        SORT = 2
    };

    enum class EnumGetPermPart
    {
        A0,
        A1,
        B0,
        B1
    };

    enum class EnumGetLen
    {
        ConsumedA,
        ConsumedB,
        OutputLen0,
        OutputLen1,
        OutputLenSum,
    };

    enum class EnumGetPred
    {
        A0,
        A1,
        B0,
        B1,
        ConsumeA,
        ConsumeB,
        Out0,
        Out1
    };

    enum class EnumPrefetch
    {
        SV_PLDL1KEEP = 0,
        SV_PLDL1STRM = 1,
        SV_PLDL2KEEP = 2,
        SV_PLDL2STRM = 3,
        SV_PLDL3KEEP = 4,
        SV_PLDL3STRM = 5,
        SV_PSTL1KEEP = 8,
        SV_PSTL1STRM = 9,
        SV_PSTL2KEEP = 10,
        SV_PSTL2STRM = 11,
        SV_PSTL3KEEP = 12,
        SV_PSTL3STRM = 13
    };
}