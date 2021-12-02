#include "joinTemplate.hpp"
#include "sparseTensor.hpp"
#include <unordered_set>
#include "../SimpleBenchmarking/BenchMarking.hpp"
#include "old.hpp"
#include "oldSIMD.hpp"

#define TENSOR_MAX_MODE 3

int main(int argc, char **argv_raw){

    std::vector<std::string> argv(argv_raw,argv_raw+argc);

    std::unordered_set<std::string> kernelSet(argv.begin(),argv.end());

    std::string fileName;
    int warmup, scalar, simd;
    warmup = scalar = simd = 1;

    for(int i=0; i<argc; ++i){
        auto &s = argv[i];
        if(s == "file") fileName = argv[i+1];
        // if(s == "kernel") kernel = argv[i+1];
        if(s == "warmup") warmup = std::stoi(argv[i+1]);
        if(s == "scalar") scalar = std::stoi(argv[i+1]);
        if(s == "simd") simd = std::stoi(argv[i+1]);
    }

    std::ifstream infile;
    infile.open(fileName,std::ios::binary);
    if(!infile){
        printf("Fail to open file %s, exit !\n",fileName.c_str());
        return -1;
    }
    infile.exceptions(std::ios::badbit | std::ios::goodbit | std::ios::eofbit);

    SparseTensor sp1,sp2;

    sp1 = loadSparseTensor(infile);
    sp1.printMeta();
    sp2 = loadSparseTensor(infile);
    sp2.printMeta();

    Timmer t;










    if(kernelSet.contains("Add")){
        std::string kernel("Add");

        printf("Executing kernel: %s\n",kernel.c_str());

        int nnzMax = sp1.nnz + sp2.nnz;

        SparseTensor sp3_scalar, sp3_simd;
        sp3_scalar.resize(sp1.mode,sp1.valNum,nnzMax);
        sp3_simd.resize(sp1.mode,sp1.valNum,nnzMax);

        // SparseTensor sp3_scalar_old, sp3_simd_old;
        // sp3_scalar_old.resize(sp1.mode,sp1.valNum,nnzMax);
        // sp3_simd_old.resize(sp1.mode,sp1.valNum,nnzMax);



        int nnzC_simd, nnzC_scalar;
        // int nnzC_simd_old, nnzC_scalar_old;

        auto foo = [&] <std::size_t...Is>(std::index_sequence<Is...>){
            if(sp1.mode > TENSOR_MAX_MODE){
                printf("Tensor mode %d exceed the largest supported tensor mode %d! Fail! \n", sp1.mode, TENSOR_MAX_MODE);
                printf("Consider change the macro definition TENSOR_MAX_MODE to larger value and recompile!\n");
                exit(-1);
            }
            (
                (
                    (sp1.mode == Is) ? (
                            t.measure("Add Scalar",warmup,scalar,[&](){
                                nnzC_scalar = sparseTensorAdd(
                                    Dummy<ImplSelect,ImplSelect::Scalar>{},
                                    Dummy<int,Is>{},
                                    sp1.data_coords(), sp1.val_coords()[0], sp1.nnz,
                                    sp2.data_coords(), sp2.val_coords()[0], sp2.nnz,
                                    sp3_scalar.data_coords(), sp3_scalar.val_coords()[0]
                                );
                            }),

                            t.measure("Add SIMD",warmup,simd,[&](){
                                nnzC_simd = sparseTensorAdd(
                                    Dummy<ImplSelect,ImplSelect::SIMD>{},
                                    Dummy<int,Is>{},
                                    sp1.data_coords(), sp1.val_coords()[0], sp1.nnz,
                                    sp2.data_coords(), sp2.val_coords()[0], sp2.nnz,
                                    sp3_simd.data_coords(), sp3_simd.val_coords()[0]
                                );
                            }),


                            // ((sp1.mode == 1) ? (
                            //         t.measure("Add Scalar old",warmup,scalar,[&](){
                            //         nnzC_scalar_old = sparse_add(
                            //             sp1.data_coords()[0],sp1.val_coords()[0], sp1.nnz,
                            //             sp2.data_coords()[0],sp2.val_coords()[0], sp2.nnz,
                            //             sp3_scalar_old.data_coords()[0], sp3_scalar_old.val_coords()[0]
                            //         );
                            //         }),

                            //         t.measure("Add SIMD old",warmup,simd,[&](){
                            //         nnzC_scalar_old = longAdd(
                            //             sp1.data_coords()[0],sp1.val_coords()[0], sp1.nnz,
                            //             sp2.data_coords()[0],sp2.val_coords()[0], sp2.nnz,
                            //             sp3_simd_old.data_coords()[0], sp3_simd_old.val_coords()[0]
                            //         );
                            //         }),

                            //         0
                            //     ) : 0
                            // ),


                            0
                    ) : 0
                ),...
            );
        };

        foo(std::make_index_sequence<TENSOR_MAX_MODE+1>{});
        sp3_simd.resizeNNZ(nnzC_simd);
        sp3_scalar.resizeNNZ(nnzC_scalar);

        // sp3_simd_old.resizeNNZ(nnzC_simd);
        // sp3_scalar_old.resizeNNZ(nnzC_scalar);

        if(sp3_scalar == sp3_simd
            // and sp3_scalar == sp3_scalar_old
            // and sp3_scalar == sp3_simd_old
            ){
            printf("Kernel %s: Result match!\n",kernel.c_str());
            sp3_scalar.printMeta();
        } else {
            printf("Kernel %s: Result misMatch! Fail! \n",kernel.c_str());
            return -1;
        }
    }











    if (kernelSet.contains("Mul")){
        std::string kernel("Mul");

        printf("Executing kernel: %s\n",kernel.c_str());

        int nnzMax = std::max(sp1.nnz,sp2.nnz);

        SparseTensor sp3_scalar, sp3_simd;
        sp3_scalar.resize(sp1.mode,sp1.valNum,nnzMax);
        sp3_simd.resize(sp1.mode,sp1.valNum,nnzMax);

        // SparseTensor sp3_scalar_old, sp3_simd_old;
        // sp3_scalar_old.resize(sp1.mode,sp1.valNum,nnzMax);
        // sp3_simd_old.resize(sp1.mode,sp1.valNum,nnzMax);



        int nnzC_simd, nnzC_scalar;
        int nnzC_simd_old, nnzC_scalar_old;

        auto foo = [&] <std::size_t...Is>(std::index_sequence<Is...>){
            if(sp1.mode > TENSOR_MAX_MODE){
                printf("Tensor mode %d exceed the largest supported tensor mode %d! Fail! \n", sp1.mode, TENSOR_MAX_MODE);
                printf("Consider change the macro definition TENSOR_MAX_MODE to larger value and recompile!\n");
                exit(-1);
            }
            (
                (
                    (sp1.mode == Is) ? (
                            t.measure("Mul Scalar",warmup,scalar,[&](){
                                nnzC_scalar = sparseTensorMul(
                                    Dummy<ImplSelect,ImplSelect::Scalar>{},
                                    Dummy<int,Is>{},
                                    sp1.data_coords(), sp1.val_coords()[0], sp1.nnz,
                                    sp2.data_coords(), sp2.val_coords()[0], sp2.nnz,
                                    sp3_scalar.data_coords(), sp3_scalar.val_coords()[0]
                                );
                            }),

                            t.measure("Mul SIMD",warmup,simd,[&](){
                                nnzC_simd = sparseTensorMul(
                                    Dummy<ImplSelect,ImplSelect::SIMD>{},
                                    Dummy<int,Is>{},
                                    sp1.data_coords(), sp1.val_coords()[0], sp1.nnz,
                                    sp2.data_coords(), sp2.val_coords()[0], sp2.nnz,
                                    sp3_simd.data_coords(), sp3_simd.val_coords()[0]
                                );
                            }),


                            // ((sp1.mode == 1) ? (
                            //         t.measure("Mul Scalar old",warmup,scalar,[&](){
                            //         nnzC_scalar_old = sparse_mul(
                            //             sp1.data_coords()[0],sp1.val_coords()[0], sp1.nnz,
                            //             sp2.data_coords()[0],sp2.val_coords()[0], sp2.nnz,
                            //             sp3_scalar_old.data_coords()[0], sp3_scalar_old.val_coords()[0]
                            //         );
                            //         }),

                            //         t.measure("Mul SIMD old",warmup,simd,[&](){
                            //         nnzC_scalar_old = longMul(
                            //             sp1.data_coords()[0],sp1.val_coords()[0], sp1.nnz,
                            //             sp2.data_coords()[0],sp2.val_coords()[0], sp2.nnz,
                            //             sp3_simd_old.data_coords()[0], sp3_simd_old.val_coords()[0]
                            //         );
                            //         }),

                            //         0
                            //     ) : 0
                            // ),


                            0
                    ) : 0
                ),...
            );
        };

        foo(std::make_index_sequence<TENSOR_MAX_MODE+1>{});
        sp3_simd.resizeNNZ(nnzC_simd);
        sp3_scalar.resizeNNZ(nnzC_scalar);

        // sp3_simd_old.resizeNNZ(nnzC_simd);
        // sp3_scalar_old.resizeNNZ(nnzC_scalar);

        if(sp3_scalar == sp3_simd
            // and sp3_scalar == sp3_scalar_old
            // and sp3_scalar == sp3_simd_old
            ){
            printf("Kernel %s: Result match!\n",kernel.c_str());
            sp3_scalar.printMeta();
        } else {
            printf("Kernel %s: Result misMatch! Fail! \n",kernel.c_str());
            return -1;
        }


    }










    if (kernelSet.contains("MulComplex")){
        if(sp1.valNum != 2 or sp2.valNum != 2){
            printf("The tensor should have 2 (real + im) value parts!");
            return -1;
        }

   std::string kernel("MulComplex");

        printf("Executing kernel: %s\n",kernel.c_str());

        int nnzMax = std::max(sp1.nnz,sp2.nnz);

        SparseTensor sp3_scalar, sp3_simd;
        sp3_scalar.resize(sp1.mode,sp1.valNum,nnzMax);
        sp3_simd.resize(sp1.mode,sp1.valNum,nnzMax);


        int nnzC_simd, nnzC_scalar;
        int nnzC_simd_old, nnzC_scalar_old;

        auto foo = [&] <std::size_t...Is>(std::index_sequence<Is...>){
            if(sp1.mode > TENSOR_MAX_MODE){
                printf("Tensor mode %d exceed the largest supported tensor mode %d! Fail! \n", sp1.mode, TENSOR_MAX_MODE);
                printf("Consider change the macro definition TENSOR_MAX_MODE to larger value and recompile!\n");
                exit(-1);
            }
            (
                (
                    (sp1.mode == Is) ? (
                            t.measure("Mul ReIm Scalar",warmup,scalar,[&](){
                                nnzC_scalar = sparseTensorMulComplex(
                                    Dummy<ImplSelect,ImplSelect::Scalar>{},
                                    Dummy<int,Is>{},
                                    sp1.data_coords(), {sp1.val_coords()[0],sp1.val_coords()[1]}, sp1.nnz,
                                    sp2.data_coords(), {sp2.val_coords()[0],sp2.val_coords()[1]}, sp2.nnz,
                                    sp3_scalar.data_coords(), {sp3_scalar.val_coords()[0],sp3_scalar.val_coords()[1]}
                                );
                            }),

                            t.measure("Mul ReIm SIMD",warmup,simd,[&](){
                                nnzC_simd = sparseTensorMulComplex(
                                    Dummy<ImplSelect,ImplSelect::SIMD>{},
                                    Dummy<int,Is>{},
                                    sp1.data_coords(), {sp1.val_coords()[0],sp1.val_coords()[1]}, sp1.nnz,
                                    sp2.data_coords(), {sp2.val_coords()[0],sp2.val_coords()[1]}, sp2.nnz,
                                    sp3_simd.data_coords(), {sp3_simd.val_coords()[0],sp3_simd.val_coords()[1]}
                                );
                            }),

                            0
                    ) : 0
                ),...
            );
        };

        foo(std::make_index_sequence<TENSOR_MAX_MODE+1>{});
        sp3_simd.resizeNNZ(nnzC_simd);
        sp3_scalar.resizeNNZ(nnzC_scalar);





        if(sp3_scalar == sp3_simd){
            printf("Kernel %s: Result match!\n",kernel.c_str());
            sp3_scalar.printMeta();
        } else {
            printf("Kernel %s: Result misMatch! Fail! \n",kernel.c_str());
            sp3_scalar.printMeta();
            sp3_simd.printMeta();
            sp3_scalar.reportDifference(sp3_simd);
            int i;
            for(i=0; i<nnzC_scalar; ++i){
                // if(sp3_scalar.vals[0][i] != sp3_simd.vals[0][i]){
                //     printf("Re not equal\n");
                //     break;
                // }
                // if(sp3_scalar.vals[1][i] != sp3_simd.vals[1][i]){
                //     printf("Im not equal\n");
                //     break;
                // }
                if(!approximateEqual(sp3_scalar.vals[0][i],sp3_simd.vals[0][i])){
                    printf("Re not equal\n");
                    break;
                }
                if(!approximateEqual(sp3_scalar.vals[1][i],sp3_simd.vals[1][i])){
                    printf("Im not equal\n");
                    break;
                }
            }
            printf("Not equal at i=%4d\n",i);
            int j=i;
            for(int i=j-5; i<j+5; ++i){
                printf("%3d\t(%4f,%4f)\t(%4f,%4f)\t(%4f,%4f)\t(%s,%s)\n",i,sp3_scalar.vals[0][i],sp3_scalar.vals[1][i],sp3_simd.vals[0][i],sp3_simd.vals[1][i],
                    sp3_simd.vals[0][i] - sp3_scalar.vals[0][i], sp3_simd.vals[1][i] - sp3_scalar.vals[1][i],
                    approximateEqual(sp3_scalar.vals[0][i],sp3_simd.vals[0][i]) ? "Y":"N",
                    approximateEqual(sp3_scalar.vals[1][i],sp3_simd.vals[1][i]) ? "Y":"N");
            }
            return -1;
        }


    }
















    if (kernelSet.contains("ShortestPath")){
        if(sp1.mode != 1 or sp2.mode != 1){
            printf("The tensor should have only 1 array for coords!");
            return -1;
        }

 std::string kernel("ShortestPath");

        printf("Executing kernel: %s\n",kernel.c_str());

        int nnzMax = sp1.nnz + sp2.nnz;

        SparseTensor sp3_scalar, sp3_simd;
        sp3_scalar.resize(sp1.mode,sp1.valNum,nnzMax);
        sp3_simd.resize(sp1.mode,sp1.valNum,nnzMax);


        // Strictly speaking, the outputs include an extra array for this oepration, so not a sparse tensor
        std::vector<int> sp3_node_scalar, sp3_node_simd;
        sp3_node_scalar.reserve(by64(nnzMax));
        sp3_node_scalar.resize(nnzMax);
        sp3_node_simd.reserve(by64(nnzMax));
        sp3_node_simd.resize(nnzMax);




        int nnzC_simd, nnzC_scalar;
        int nnzC_simd_old, nnzC_scalar_old;

 
        if(sp1.mode > TENSOR_MAX_MODE){
            printf("Tensor mode %d exceed the largest supported tensor mode %d! Fail! \n", sp1.mode, TENSOR_MAX_MODE);
            printf("Consider change the macro definition TENSOR_MAX_MODE to larger value and recompile!\n");
            exit(-1);
        }

        t.measure("ShortPath Scalar",warmup,scalar,[&](){
            nnzC_scalar = selectSorterPath(
                Dummy<ImplSelect,ImplSelect::Scalar>{},
                sp1.data_coords()[0], sp1.val_coords()[0], sp1.nnz,
                sp2.data_coords()[0], sp2.val_coords()[0], sp2.nnz,
                sp3_scalar.data_coords()[0],sp3_node_scalar.data(), sp3_scalar.val_coords()[0],
                0,0.0,0,0.0
            );
        });

        t.measure("ShortPath SIMD",warmup,simd,[&](){
            nnzC_simd = selectSorterPath(
                Dummy<ImplSelect,ImplSelect::SIMD>{},
                sp1.data_coords()[0], sp1.val_coords()[0], sp1.nnz,
                sp2.data_coords()[0], sp2.val_coords()[0], sp2.nnz,
                sp3_simd.data_coords()[0],sp3_node_simd.data(), sp3_simd.val_coords()[0],
                0,0.0,0,0.0
            );
        });

        sp3_scalar.resizeNNZ(nnzC_scalar);
        sp3_node_scalar.resize(nnzC_scalar);
        sp3_simd.resizeNNZ(nnzC_simd);
        sp3_node_simd.resize(nnzC_simd);

        if(sp3_scalar == sp3_simd){
            printf("Kernel %s: Result match!\n",kernel.c_str());
            sp3_scalar.printMeta();
        } else {
            printf("Kernel %s: Result misMatch! Fail! \n",kernel.c_str());
            return -1;
        }

    }
    printf("Pass!\n");
    t.dump();
    return 0;



    // std::vector<int> sa = {1,2,3,4};
    // std::vector<float> sav = {1,1,1,1};
    // std::vector<float> sav_im = {1,1,1,1};
    // std::vector<int> sb = {2,4,6,8};
    // std::vector<float> sbv = {1,1,1,1};
    // std::vector<float> sbv_im = {1,1,1,1};

    // int lenA = 800;
    // int lenB = 800;
    // sa.resize(lenA); sav.resize(lenA); sav_im.resize(lenA);
    // sb.resize(lenB); sbv.resize(lenB); sbv_im.resize(lenB);
    // for(int i=0; i<lenA; ++i){
    //     sa[i] = i;
    //     sav[i] = 2;
    //     sav_im[i] = 3;
    //     // sav[i] = i*i;
    // }
    // for(int i=0; i<lenB; ++i){
    //     sb[i] = i*2;
    //     sbv[i] = 1;
    //     sbv_im[i] = 2;
    // }

    // std::vector<int> sc(sa.size()+sb.size(),0);
    // std::vector<int> scNode(sa.size()+sb.size(),0);
    // std::vector<float> scv(sa.size()+sb.size(),0);
    // std::vector<float> scv_im(sa.size()+sb.size(),0);
    // std::vector<int> sc_vec(sa.size()+sb.size(),0);
    // std::vector<int> scNode_vec(sa.size()+sb.size(),0);
    // std::vector<float> scv_vec(sa.size()+sb.size(),0);
    // std::vector<float> scv_im_vec(sa.size()+sb.size(),0);
    // printf("sc size = %lu\n",sc.size());

    // // int lc, lc_vec;
    // // lc = lc_vec = 0;

    // int lc = selectSorterPath(
    //         Dummy<ImplSelect,ImplSelect::Scalar>{},
    //         sa.data(),sav.data(),int(sa.size()),
    //         sb.data(),sbv.data(),int(sb.size()),
    //         sc.data(),scNode.data(),scv.data(),
    //         111,0,
    //         222,0
    //         );
    // int lc_vec = selectSorterPath(
    //         Dummy<ImplSelect,ImplSelect::SIMD>{},
    //         sa.data(),sav.data(),int(sa.size()),
    //         sb.data(),sbv.data(),int(sb.size()),
    //         sc_vec.data(),scNode_vec.data(),scv_vec.data(),
    //         111,0,
    //         222,0
    //         );


    // int lc_vec = sparseTensorAdd(
    //     Dummy<ImplSelect,ImplSelect::SIMD>{},
    //     Dummy<int,1>{},
    //     std::vector<int*>{sa.data()},sav.data(),int(sa.size()),
    //     std::vector<int*>{sb.data()},sbv.data(),int(sb.size()),
    //     std::vector<int*>{sc_vec.data()},scv_vec.data()
    // );

    // int lc = sparseTensorAdd(
    //     Dummy<ImplSelect,ImplSelect::Scalar>{},
    //     Dummy<int,1>{},
    //     std::vector<int*>{sa.data()},sav.data(),int(sa.size()),
    //     std::vector<int*>{sb.data()},sbv.data(),int(sb.size()),
    //     std::vector<int*>{sc.data()},scv.data()
    // );
    // int lc_vec = sparseTensorMul(
    //     Dummy<ImplSelect,ImplSelect::SIMD>{},
    //     Dummy<int,1>{},
    //     {sa.data()},sav.data(),int(sa.size()),
    //     {sb.data()},sbv.data(),int(sb.size()),
    //     {sc_vec.data()},scv_vec.data()
    // );

    // int lc = sparseTensorMul(
    //     Dummy<ImplSelect,ImplSelect::Scalar>{},
    //     Dummy<int,1>{},
    //     {sa.data()},sav.data(),int(sa.size()),
    //     {sb.data()},sbv.data(),int(sb.size()),
    //     {sc.data()},scv.data()
    // );
    // int lc_vec = sparseTensorMulComplex(
    //     Dummy<ImplSelect,ImplSelect::SIMD>{},
    //     Dummy<int,1>{},
    //     {sa.data()},{sav.data(),sav_im.data()},int(sa.size()),
    //     {sb.data()},{sbv.data(),sbv_im.data()},int(sb.size()),
    //     {sc_vec.data()},{scv_vec.data(),scv_im_vec.data()}
    // );

    // int lc = sparseTensorMulComplex(
    //     Dummy<ImplSelect,ImplSelect::Scalar>{},
    //     Dummy<int,1>{},
    //     {sa.data()},{sav.data(),sav_im.data()},int(sa.size()),
    //     {sb.data()},{sbv.data(),sbv_im.data()},int(sb.size()),
    //     {sc.data()},{scv.data(),scv_im.data()}
    // );



    // for(int i=0; i<10;++i){
    //     printf("%3d:\t%4d\t%4d\t%4f\n",i,sc[i],scNode[i],scv[i]);
    // }

    // bool pass = true;
    // printf("lc = %d, lc_vec = %d", lc, lc_vec);
    // if(lc == lc_vec) printf("Length equal!\n"); else pass = false;
    // if(std::equal(sc.begin(),sc.begin()+lc, sc_vec.begin())) printf("Key equal!\n"); else pass = false;
    // if(std::equal(scNode.begin(),scNode.begin()+lc, scNode_vec.begin())) printf("Node equal!\n"); else pass = false;
    // if(std::equal(scv.begin(),scv.begin()+lc, scv_vec.begin())) printf("Value equal!\n"); else pass = false;

    // if(pass){
    //     printf("/////////// Final result: Pass !!!\n");
    // } else {
    //     printf("XXXXXXXX Final result: Fail !!!\n");
    // }

    
}