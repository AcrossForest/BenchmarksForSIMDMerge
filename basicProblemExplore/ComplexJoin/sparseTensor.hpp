#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <algorithm>
#include <stdlib.h>
#include <limits>
inline std::size_t by64(std::size_t sz){
  return (sz/64) * 64 + (sz%64 != 0 ?64:0);
}

template<class T>
inline bool approximateEqual(T va, T vb){
    if constexpr (std::numeric_limits<T>::is_integer) {
        return va==vb;
    } else {
        constexpr T epsilon = 10* std::numeric_limits<T>::epsilon();
        return (va - vb) < epsilon && (vb - va) < epsilon;
    }
}

template<class T>
inline bool approximateEqual(const std::vector<T>& va, const std::vector<T>& vb){
    if constexpr (std::numeric_limits<T>::is_integer) {
        return va==vb;
    } else {
        if(va.size() != vb.size()) return false;
        bool equal = true;
        // auto epsilon = std::numeric_limits<T>::epsilon();
        for(std::size_t i=0; i<va.size(); ++i){
            equal &= approximateEqual(va[i],vb[i]);// (va[i] - vb[i]) < epsilon && (vb[i] - va[i]) < epsilon;
        }
        return equal;
    }
}

template<class T>
inline bool approximateEqual(const std::vector<std::vector<T>>& va, const std::vector<std::vector<T>>& vb){
    if(va.size() != vb.size()) return false;
    bool equal =true;
    for(std::size_t i=0; i<va.size(); ++i){
        equal &= approximateEqual(va[i],vb[i]);
    }
    return equal;
}

struct SparseTensor{
    int mode;
    int valNum;
    int nnz;
    std::vector<std::vector<int>> coors;
    std::vector<std::vector<float>> vals;

    void resize(int _mode, int _valNum, int _nnz){
        mode = _mode;
        valNum = _valNum;
        nnz = _nnz;
        coors.resize(mode);
        vals.resize(valNum);
        resizeNNZ(_nnz);
    }
    void resizeNNZ(int _nnz){
        nnz = _nnz;
        for(auto &cr:coors){
            cr.reserve(by64(_nnz));
            cr.resize(_nnz);
        }
        for(auto &vr:vals){
            vr.reserve(by64(_nnz));
            vr.resize(_nnz);
        }
    }

    std::vector<int*> data_coords(){
        std::vector<int*> ptrs(mode);
        for(int i=0; i<mode;++i){
            ptrs[i] = coors[i].data();
        }
        return ptrs;
    }
    std::vector<float*> val_coords(){
        std::vector<float*> ptrs(valNum);
        for(int i=0; i<valNum;++i){
            ptrs[i] = vals[i].data();
        }
        return ptrs;
    }

    void reportDifference(const SparseTensor& other) const {
        if(mode != other.mode){
            printf("Reason: Mode not equal\n");
        }
        if(valNum != other.valNum){
            printf("Reason: valNum not equal\n");
        }
        if(approximateEqual(coors, other.coors)){
            printf("Coords not equal\n");
        }
        if(approximateEqual(vals, other.vals)){
            printf("Vals not equal\n");
        }
        // if(coors != other.coors){
        //     printf("Coords not equal\n");
        // }
        // if(vals != other.vals){
        //     printf("Vals not equal\n");
        // }
    }


    bool operator==(const SparseTensor& other) const {
        return  mode==other.mode 
            && valNum == other.valNum
            && nnz == other.nnz
            && approximateEqual(coors, other.coors)
            && approximateEqual(vals, other.vals);
            // && coors == other.coors
            // && vals == other.vals;
    }

    void printMeta(std::string name="") const {
        printf("Tensor %10s: mode = %2d, valNum = %2d, nnz = %6d\n",name.c_str(),mode,valNum,nnz);
    }
};

inline SparseTensor loadSparseTensor(std::istream& file){
    int magicNum = 0;
    file.read((char*)&magicNum,sizeof(int));
    int mode;
    if(magicNum!=741){
        int position = file.tellg();
        printf("Magic number is not 741 (get %d)! file position = %d, Exit! \n",magicNum, position);
        exit(-1);
    }
    int valNum;
    int nnz;
    file.read((char*)&mode,sizeof(int));
    file.read((char*)&valNum,sizeof(int));
    file.read((char*)&nnz,sizeof(int));
    SparseTensor sp;

    sp.resize(mode,valNum,nnz);
    for(auto &cr:sp.coors){
        file.read((char*)cr.data(),nnz*sizeof(int));
        // bool check = std::is_sorted(cr.begin(),cr.begin() + nnz);
        // if(!check){
        //     printf("Error! The loaded index sequence is not sorted! Exit!\n");
        //     exit(-1);
        // }
    }

    for(int i=1; i<nnz; ++i){
        auto cmp = sp.coors[0][i-1] <=> sp.coors[0][i];
        for(int m=1; m < mode; ++m){
            cmp = std::is_eq(cmp) ? (sp.coors[m][i-1] <=> sp.coors[m][i]) : cmp;
        }
        if(!std::is_lt(cmp)){
            printf("Error! The loaded index sequence is not sorted! Exit!\n");
            exit(-1);
        }
    }
    for(auto &vr:sp.vals){
        file.read((char*)vr.data(),nnz*sizeof(float));
    }
    return sp;
}