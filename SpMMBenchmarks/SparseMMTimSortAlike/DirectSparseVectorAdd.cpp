
#include "SparseMatTool/format.hpp"
#include "VectorUtils.hpp"
Size_t directVectorAdd(Idx *__restrict a_pidx, Val *__restrict a_pval, Size_t a_len,
              Idx *__restrict b_pidx, Val *__restrict b_pval, Size_t b_len,
              Idx *__restrict c_pidx, Val *__restrict c_pval)
// Size_t merger(Idx * a_pidx, Val * a_pval, Size_t a_len,
//               Idx * b_pidx, Val * b_pval, Size_t b_len,
//               Idx * c_pidx, Val * c_pval)
{
    Size_t a_used = 0;
    Size_t b_used = 0;
    Size_t c_filled = 0;

    if (a_len > 0 && b_len > 0)
    {
        // Although ideally we can use "load in advance" to reduce
        // the potential critical path and reduce memory load
        // But if in x86 register spilling happens anyway ....
        Idx a_this_idx = a_pidx[0];
        Idx b_this_idx = b_pidx[0];
        Val a_this_val = a_pval[0];
        Val b_this_val = b_pval[0];
        while (a_used < a_len && b_used < b_len)
        {
            Val c_this_val_a = 0;
            Val c_this_val_b = 0;
            Idx c_this_idx = std::min(a_this_idx, b_this_idx);
            if (a_this_idx == c_this_idx)
            {
                a_used++;
                c_this_val_a = a_this_val;
                a_this_val = a_pval[a_used]; // Potential Issue: out of range error
                a_this_idx = a_pidx[a_used]; // Although it might not be used
            }

            if (b_this_idx == c_this_idx)
            {
                b_used++;
                c_this_val_b = b_this_val;
                b_this_idx = b_pidx[b_used];
                b_this_val = b_pval[b_used];
            }

            c_pidx[c_filled] = c_this_idx;
            c_pval[c_filled] = c_this_val_a + c_this_val_b;
            c_filled++;
        }
    }
    if (a_used < a_len)
    {
        vectorCopy(
            a_pidx+a_used, a_pval+ a_used, a_len - a_used,
            c_pidx + c_filled, c_pval + c_filled
        );
        // std::copy(a_pidx + a_used, a_pidx + a_len, c_pidx + c_filled);
        // std::copy(a_pval + a_used, a_pval + a_len, c_pval + c_filled);
        c_filled += a_len - a_used;
        a_used = a_len;
    }
    if (b_used < b_len)
    {
        vectorCopy(
            b_pidx+b_used, b_pval+ b_used, b_len - b_used,
            c_pidx + c_filled, c_pval + c_filled
        );
        // std::copy(b_pidx + b_used, b_pidx + b_len, c_pidx + c_filled);
        // std::copy(b_pval + b_used, b_pval + b_len, c_pval + c_filled);
        c_filled += b_len - b_used;
        b_used = b_len;
    }
    return c_filled;
}

