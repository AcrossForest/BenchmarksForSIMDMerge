#include "sequential.hpp"
#include "simdJoin.hpp"
#include <algorithm>
#include <random>
#include <unordered_set>
#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include "../SimpleBenchmarking/BenchMarking.hpp"
#include <fstream>

std::size_t by64(std::size_t sz){
  return (sz/64) * 64 + (sz%64 != 0 ?64:0);
}

std::vector<uint32_t> loadVector(std::istream& file){
  int len;
  file.read((char*)&len,sizeof(int));
  std::vector<uint32_t> vec;
  vec.resize(by64(len));

  file.read((char*)vec.data(),len * sizeof(uint32_t));
  vec.resize(len);
  return vec;
}

template<SpSpEnum::PolicyStruct T>
struct PolicyDummy{};

std::vector<uint32_t> makeUniqueRandom(int overlap, int seedOverlap, int count, int seed){
    std::minstd_rand0 overlapGen(seedOverlap);
    std::minstd_rand0 gen(seed);
    constexpr int i = sizeof(overlapGen);
    std::uniform_int_distribution<uint32_t> dist;
    std::unordered_set<uint32_t> set;
    while(set.size() < overlap){
        set.insert(dist(overlapGen));
    }
    while(set.size() < count){
        set.insert(dist(gen));
    }
    std::vector<uint> vec(set.begin(),set.end());
    std::ranges::sort(vec);
    return vec;
}

std::vector<uint32_t> makeMonotone(int count){
    std::vector<uint32_t> vec(count);
    std::iota(vec.begin(),vec.end(),0);
    return vec;
}


int main(int argc, char **argv_raw){
    std::vector<std::string> argv(argv_raw,argv_raw+argc);
    int len1,len2,overlap;
    len1 = len2 = 10000;
    overlap = 5000;
    bool generate = false;
    int scalarTimes = 1;
    int simdTimes = 1;
    std::string fileName;
    for(int i=0; i<argc; ++i){
        auto &s = argv[i];
        if(s == "len1") len1 = std::stoi(argv[i+1]);
        if(s == "len2") len2 = std::stoi(argv[i+1]);
        if(s == "overlap") overlap = std::stoi(argv[i+1]);
        if(s == "generate") generate = true;
        if(s == "file") fileName = argv[i+1];
        if(s == "scalar") scalarTimes = std::stoi(argv[i+1]);
        if(s == "simd") simdTimes = std::stoi(argv[i+1]);

    }
    Timmer t;

    std::vector<uint32_t> aidx,aval,bidx,bval;

    if(generate){
        aidx = makeUniqueRandom(overlap,0,len1,3);
        aval = makeMonotone(len1);
        bidx = makeUniqueRandom(overlap,0,len2,4);
        bval = makeMonotone(len2);
    } else {
            auto cwd = std::filesystem::current_path();
            std::cout << "Current Working Directory: " << cwd << std::endl;
            std::cout << "Attempt to open file: " << fileName << std::endl;
            std::ifstream file(fileName,std::ifstream::binary | std::ifstream::in);
            if(!file){
                std::cout << "The file does not exisit!" << std::endl;
                return 1;
            }

            aidx = loadVector(file);
            std::cout << "Load aidx: "<< aidx.size() << std::endl;;
            aval = loadVector(file);
            std::cout << "Load aval: "<< aval.size() << std::endl;;
            bidx = loadVector(file);
            std::cout << "Load bidx: "<< bidx.size() << std::endl;;
            bval = loadVector(file);
            std::cout << "Load bval: "<< bval.size() << std::endl;;
            file.close();
    }


    int maxLen = by64(len1 + len2);
    std::vector<uint32_t> cidx(maxLen);
    std::vector<uint32_t> cvala(maxLen);
    std::vector<uint32_t> cvalb(maxLen);

    std::vector<uint32_t> g_cidx(maxLen);
    std::vector<uint32_t> g_cvala(maxLen);
    std::vector<uint32_t> g_cvalb(maxLen);

    int lenc,g_lenc;
    bool allCorrect = true;
    auto check = [&](){
        bool isFalse;
        if (isFalse |= (lenc != g_lenc))
            {printf("lenc is incorrct: \tget=%8d\texpect=%8d\n",lenc,g_lenc);}
        if (isFalse |= !std::equal(g_cidx.begin(),g_cidx.begin()+g_lenc,cidx.begin()))
            {printf("idx is incorrect\n");}
        if (isFalse |= !std::equal(g_cvala.begin(),g_cvala.begin()+g_lenc,cvala.begin()))
            {printf("cvala is incorrect\n");}
        if (isFalse |= !std::equal(g_cvalb.begin(),g_cvalb.begin()+g_lenc,cvalb.begin()))
            {printf("cvalb is incorrect\n");}
        
        if (!isFalse){
            printf("Correct!\n");
        } else {
            printf("Wrong!\n");
        }
        allCorrect &= !isFalse;
        return !isFalse;
    };

    

    auto standardMeasure = [&]<SpSpEnum::PolicyStruct policy>(std::string name, JoinFunc scalarFun, PolicyDummy<policy>){
        t.measure(name + " Scalar",1,scalarTimes,[&](){
            g_lenc = scalarFun(
                aidx.data(),aval.data(),0u,aidx.size(),
                bidx.data(),bval.data(),0u,bidx.size(),
                g_cidx.data(),g_cvala.data(),g_cvalb.data()
            );    
        });
        t.measure(name + " SIMD",1,simdTimes,[&](){
            lenc = JoinOp_Unique<policy>::join(
                aidx.data(),aval.data(),0u,aidx.size(),
                bidx.data(),bval.data(),0u,bidx.size(),
                cidx.data(),cvala.data(),cvalb.data()
            );
        });
        printf("For %5s - join, g_lenc = %5d\n",name.c_str(),g_lenc);
        check();
    };

    standardMeasure("Join-Full-Outer",outerJoin,PolicyDummy<PolicyOR>{});
    standardMeasure("Join-Inner",innerJoin,PolicyDummy<PolicyAND>{});
    standardMeasure("Join-Outer-Ex",xorJoin,PolicyDummy<PolicyXOR>{});
    standardMeasure("Join-Left-Ex",diffJoin,PolicyDummy<PolicyDiff>{});
    standardMeasure("Join-Left",leftJoin,PolicyDummy<PolicyLeft>{});

    if(allCorrect){
        t.dump();
    }
}