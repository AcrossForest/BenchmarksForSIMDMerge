#include "SparseMatTool/format.hpp"
#include "SparseMMHeapAccum/SparseMMHeapAccum.hpp"
#include "SparseMMHeapAccum/heapReplace.hpp"
#include "Benchmarking/benchmarking.hpp"
#include <limits>
#include <algorithm>

CSR SpMM_Heap(TimmerHelper& timmerHelper, const CSR &a, const CSR &b)
{
    // c_{m,n} = a_{m,k} * b_{k,n}
    Idx m, k, n;
    m = a.m;
    k = a.n;
    n = b.n;

    // Estimate the size of output matrix
    Size_t nnzCMax = 0;
    NNZIdx nnzARowMax = 0;
    for (Idx r = 0; r < m; ++r)
    {
        // The nnz of C[r,:]
        Size_t nnzCr = 0;
        for (Size_t off = a.rowBeginOffset[r]; off < a.rowBeginOffset[r + 1]; ++off)
        {
            Idx a_col = a.colIdx[off];
            nnzCr += b.rowBeginOffset[a_col + 1] - b.rowBeginOffset[a_col];
        }
        nnzCMax += std::min<Size_t>(nnzCr, n);
        nnzARowMax = std::max<Idx>(nnzARowMax, a.rowBeginOffset[r + 1] - a.rowBeginOffset[r]);
    }

    CSR c;
    c.m = m;
    c.n = n;
    c.rowBeginOffset.resize(m + 1);
    c.colIdx.resize(nnzCMax);
    c.values.resize(nnzCMax);
    Size_t nnzCReal = 0;

    const Idx sentinalVal = std::numeric_limits<Idx>::max();

    // Names:
    // For scalars indexes:
    // (1) a's row [m] (2) a's col / b's row[k] (3) b's col[n]
    // For dense form, v_m = vec[m]
    // For sparse form, v_col[m_c] = vec[m_c]. suffix '_c' means compressed
    // Type: Idx idx_m,idx_k,idx_n;
    // Type: NNZIdx idx_k_c, idx_n_c;
    // For vectors:
    // Factor 1: It's index type, 2: It's value's type

    // a_row, (a_rowBegin,a_rowEnd) = enumerate(a.rowBeginOffset)
    // a_col_c_off, a_col_c_0, a_col, a_val = zip(range(a_rowBegin,a_rowEnd),enumerate(zip(a.colIdx,a.values)[a_rowBegin,a_rowEnd]))
    // b_row, (b_rowBegin, b_rowEnd) = a_col, b.rowBeginOffset(a_col), b.rowBeginoffset(a_col+1)
    // b_col_c_off, b_col_c_0, b_col, b_val = zip(range(a_rowBegin,a_rowEnd),enumerate(zip(b.colIdx,b.values)[b_rowBegin,b_rowEnd]))

    // Index by heapPos (A numer encoding of a tree position)
    Buffer<NNZIdx> heap(nnzARowMax);
    // Indexed by a_col_c_0
    Buffer<Idx> first_b_col(nnzARowMax);
    Buffer<Size_t> first_b_col_c_off(nnzARowMax);
    Buffer<Size_t> b_rowEnd(nnzARowMax);
    Buffer<Val> a_val(nnzARowMax);

    timmerHelper.start();

    auto compare = [&](NNZIdx a, NNZIdx b) {
        return first_b_col[a] > first_b_col[b];
    };

    auto getfirst_b_col = [&](NNZIdx a_col_c_0) {
        return first_b_col_c_off[a_col_c_0] < b_rowEnd[a_col_c_0] ? b.colIdx[first_b_col_c_off[a_col_c_0]] : sentinalVal;
    };

    for (Idx r = 0; r < m; ++r)
    {
        c.rowBeginOffset[r] = nnzCReal;

        Size_t a_rowBegin = a.rowBeginOffset[r];
        NNZIdx a_rowNNZ = a.rowBeginOffset[r + 1] - a.rowBeginOffset[r];
        if (a_rowNNZ == 0)
            continue;
        for (NNZIdx a_col_c_0 = 0; a_col_c_0 < a_rowNNZ; ++a_col_c_0)
        {
            NNZIdx a_col_c_off = a_col_c_0 + a_rowBegin;

            heap[a_col_c_0] = a_col_c_0;
            Idx a_col = a.colIdx[a_col_c_off];
            a_val[a_col_c_0] = a.values[a_col_c_off];

            Idx b_row = a_col;

            first_b_col_c_off[a_col_c_0] = b.rowBeginOffset[b_row];
            b_rowEnd[a_col_c_0] = b.rowBeginOffset[b_row + 1];

            first_b_col[a_col_c_0] = getfirst_b_col(a_col_c_0);
        }

        // 1 make heap initial
        std::make_heap(heap.begin(), heap.begin() + a_rowNNZ, compare);

        // Just to makesure it is not equal to any possible index
        Idx lastSeen_b_col = -1; // Ok, lastSeen_b_col is unsigned, so round to max.
        nnzCReal -= 1;

        while (true)
        {
            // 2 Check if the heap is over (for example, empty B)
            // Only need to check the head of the heap
            NNZIdx top_a_col_c_0 = heap.front();
            if (first_b_col[top_a_col_c_0] == sentinalVal)
                break;

            // 3 use the first element of the heap
            Val fa = a_val[top_a_col_c_0];
            Val fb = b.values[first_b_col_c_off[top_a_col_c_0]];
            Val mul = fa * fb;
            Idx b_col = first_b_col[top_a_col_c_0];

            // Now, c[r][b_col] += mul
            // insert(c,r,b_col,mul)

            if (b_col != lastSeen_b_col)
            {
                nnzCReal++;
                c.colIdx[nnzCReal] = b_col;
                c.values[nnzCReal] = 0;
                lastSeen_b_col = b_col;
            }
            c.values[nnzCReal] += mul;

            // 4 Now, advanced the head
            first_b_col_c_off[top_a_col_c_0]++;
            first_b_col[top_a_col_c_0] = getfirst_b_col(top_a_col_c_0);

            // 5 Recover the head status
            // std::pop_heap(heap.begin(),heap.begin()+a_rowNNZ,compare);
            // *(heap.begin() + a_rowNNZ - 1) = top_a_col_c_0;
            // std::push_heap(heap.begin(),heap.begin()+a_rowNNZ,compare);

            NNZIdx badPos = 0;
            Idx top = first_b_col[top_a_col_c_0]; // Now it have new b_col
            while (true)
            {
                NNZIdx leftChild = 2 * badPos + 1;
                NNZIdx rightChild = 2 * badPos + 2;

                if (rightChild < a_rowNNZ)[[likely]]
                {
                    NNZIdx left_a_col_c_0, right_a_col_c_0;
                    Idx left_b_col, right_b_col;

                    left_a_col_c_0 = heap[leftChild];
                    right_a_col_c_0 = heap[rightChild];
                    left_b_col = first_b_col[left_a_col_c_0];
                    right_b_col = first_b_col[right_a_col_c_0];
                    //////////////////////////////////////////////////////
                    //
                    //      Below are performance hot spot.
                    //      Some optimization seems not working on X86, but might work on ARM/RISCV5
                    //      Maybe because X86 have very limited number of registers. So it didn't generate
                    //      conditional selection instruction as I expected (such as cmov)
                    //
                    ///////////////////////////////////////////////////
                    //                 Early break: 2 variants
                    //////////////////////////////////////////////

                    // Variant 1
                    // if (top <= left_b_col && top <= right_b_col) {
                    //     break;
                    // }

                    // Variant 2
                    auto smallerOne = (left_b_col <= right_b_col) ? left_b_col : right_b_col;
                    if (top <= smallerOne){
                        break;
                    }

                    ///////////////////////////////////////////////////
                    //                 Heap: select the correct child:  3 variants
                    //////////////////////////////////////////////

                    // Variant 1
                    NNZIdx theChild = (left_b_col <= right_b_col) ? leftChild : rightChild;
                    heap[badPos] = heap[theChild]; // Variant 1
                    badPos = theChild;

                    // Variant 2
                    // NNZIdx theChild = (left_b_col <= right_b_col) ? leftChild : rightChild;
                    // heap[badPos] =  (left_b_col <= right_b_col) ? left_a_col_c_0 : right_a_col_c_0; // Variant 2
                    // badPos = theChild;

                    // Variant 3
                    // if(left <= right){
                    //     heap[badPos] = heap[leftChild];
                    //     badPos = leftChild;
                    // } else {
                    //     heap[badPos] = heap[rightChild];
                    //     badPos = rightChild;
                    // }
                    ///////////////////////////////////////////////// END
                }
                else if (leftChild >= a_rowNNZ)
                {
                    break;
                }
                else
                { // leftChild < rightChild == a_rowNNZ
                    Idx left;
                    left = first_b_col[heap[leftChild]];
                    if (left < top)
                    {
                        heap[badPos] = heap[leftChild];
                        badPos = leftChild;
                    }
                    break;
                }
            }

            heap[badPos] = top_a_col_c_0;

            // if(!std::is_heap(heap.begin(),heap.begin()+a_rowNNZ,compare)){
            //     throw std::logic_error("It is not a heap...");
            // }

            // heapreplace(heap.begin(),heap.begin()+a_rowNNZ,compare);
        }
        nnzCReal++; // Close the added column of c
    }


    timmerHelper.end();

    c.nnz = nnzCReal;
    c.rowBeginOffset.back() = nnzCReal;
    c.colIdx.resize(nnzCReal);
    c.colIdx.shrink_to_fit();
    c.values.resize(nnzCReal);
    c.values.shrink_to_fit();

    return c;
}