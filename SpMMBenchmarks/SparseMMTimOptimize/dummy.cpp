#include "scalarSparseAdd.hpp"
#include "simdSparseAdd.hpp"
#include "rowStack.hpp"
#include "sparseGemm.hpp"
#include <memory>

// int test1(){
//     uint32_t bidx[] = {1,2,1,3,1,4,1,5,1,6,1,7,1,8};
//     float bval[] =    {1,1,1,1,1,1,1,1,1,1,1,1,1,1};
//     auto rowStack = SparseRowStack(SIMDToolkit{});
//     rowStack.reserve(100,bidx,bval);
//     Timmer t;
//     auto x = t.AllocateTimmerHelper("ff");
//     SpMM_TimSortOptimized(x,ScalarToolkit{},CSR{}, CSR{});

//     float base = 1;
//     for(int i=0; i<7; ++i){
//         int init = rowStack.getStackSize();
//         rowStack.putOnTop(base,2*i,2);
//         base *= 10;
//         int beforeReduce = rowStack.getStackSize();
//         rowStack.reduce();
//         printf("Stack size=%4d->%4d->%4d\n",init,beforeReduce,rowStack.getStackSize());
//         // Now print the how stack
//         printf(" \n\n\n//////////////// Print stack:\n");
//         for(int i=0; i<rowStack.getStackSize(); ++i){
//             printf("[%4d][%4d]\t",i,rowStack.getTop(i).length);
//             auto info = rowStack.getTop(i);
//             for(int j=0; j<rowStack.getTop(i).length; ++j){
//                 printf("%2d\t",
//                     rowStack.idxBuffer[info.pos][info.start + j]
//                 );
//             }
//             printf("\n");
//         }

//     }
//     rowStack.reduce(true,2);
//         printf(" \n\n\n//////////////// Print stack:\n");
//         for(int i=0; i<rowStack.getStackSize(); ++i){
//             printf("[%4d][%4d]\t",i,rowStack.getTop(i).length);
//             auto info = rowStack.getTop(i);
//             for(int j=0; j<rowStack.getTop(i).length; ++j){
//                 printf("%2d\t",
//                     rowStack.idxBuffer[info.pos][info.start + j]
//                 );
//             }
//             printf("\n");
//         }

//     auto c1idx = std::make_unique<uint32_t[]>(100);
//     auto c1val = std::make_unique<float[]>(100);

//     int c = rowStack.exportResultTo(c1idx.get(),c1val.get());
//     printf("c = %d\n",c);
//     for(int i=0; i<c; ++i){
//         printf("%5d\t%5f\n",c1idx[i],c1val[i]);
//     }

// }

// int test1(){
//     auto aidx = std::make_unique<uint32_t[]>(100);
//     auto bidx = std::make_unique<uint32_t[]>(100);
//     auto aval = std::make_unique<float[]>(100);
//     auto bval = std::make_unique<float[]>(100);

//     auto c1idx = std::make_unique<uint32_t[]>(100);
//     auto c1val = std::make_unique<float[]>(100);
//     auto c2idx = std::make_unique<uint32_t[]>(100);
//     auto c2val = std::make_unique<float[]>(100);


//     float sa = 3, sb = 4;
//     for(int i=0; i<20; ++i){
//         aidx[i] = i;
//         bidx[i] = 2*i;
//         aval[i] = i;
//         bval[i] = i*i;

//     }


 
//     int c1 =SIMDToolkit::sparseAdd(
//         SIMDToolkit::mkMulOp(sa),
//         SIMDToolkit::mkMulOp(sb),
//         aidx.get(),aval.get(),20,
//         bidx.get(),bval.get(),20,
//         c1idx.get(),c1val.get()
//     );
//     int c2 =ScalarToolkit::sparseAdd(
//         ScalarToolkit::mkMulOp(sa),
//         ScalarToolkit::mkMulOp(sb),
//         aidx.get(),aval.get(),20,
//         bidx.get(),bval.get(),20,
//         c2idx.get(),c2val.get()
//     );

//     if(c1 != c2){
//         printf("c1 != c2\n");
//         return -1;
//     }

//     for(int i=0; i<c1; ++i){
//         if(c1idx[i] != c2idx[i]){
//             printf("c1idx[%d] != c2idx[%d] (%d != %d)!\n",i,i,c1idx[i],c2idx[i]);
//             return -1;
//         }
//         if(c1val[i] != c2val[i]){
//             printf("c1val[%d] != c2val[%d] (%f != %f)!\n",i,i,c1val[i],c2val[i]);
//             return -1;
//         }
//     }
//     printf("Correct! c1=%d, c2=%d\n",c1,c2);

//     return 0;
// }