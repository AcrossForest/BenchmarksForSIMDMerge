#include <iostream>
#include <fstream>
#include <filesystem>
#include "../SimpleBenchmarking/BenchMarking.hpp"
#include "Sequential.hpp"
#include "Simd.hpp"
#include <algorithm>

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


int main(int argc, char **argv_raw){
  std::vector<std::string> argv(argv_raw,argv_raw+argc);
    int scalarTimes = 1;
    int simdTimes = 1;
    std::string fileName;
    for(int i=0; i<argc; ++i){
        auto &s = argv[i];
        if(s == "file") fileName = argv[i+1];
        if(s == "scalar") scalarTimes = std::stoi(argv[i+1]);
        if(s == "simd") simdTimes = std::stoi(argv[i+1]);

  }
  // std::cout << "Begin of program" << std::endl;
  auto cwd = std::filesystem::current_path();
  std::ifstream file(fileName,std::ifstream::binary | std::ifstream::in);
  if(!file){
    std::cout << "The file does not exisit!" << std::endl;
    return 1;
  }

  auto a = loadVector(file);
  // printf("What is the fuck?");
  std::cout << "Load a: "<< a.size() << std::endl;;
  auto b = loadVector(file);
  std::cout << "Load b: "<< b.size() << std::endl;
  file.close();


  int lenC1,lenC2;
  std::vector<uint32_t> c1(by64(a.size()+b.size()));
  std::vector<uint32_t> c2(by64(a.size()+b.size()));

  Timmer t;


  t.measure("Union Scalar",1,scalarTimes,[&](){
    lenC1 = findUnion(a.data(),a.size(),b.data(),b.size(),c1.data());
  });
  t.measure("Union SIMD",1,simdTimes,[&](){
    lenC2 = SetOp<PolicyOR>::op(a.data(),a.size(),b.data(),b.size(),c2.data());
  });

  if(lenC1 == lenC2 && std::equal(c1.data(),c1.data()+lenC1,c2.data())){
    std::cout << "Result match" << std::endl;
  } else {
    std::cout << "Result don't match" << std::endl;
    return -1;
  }

  t.measure("Intersect Scalar",1,scalarTimes,[&](){
    lenC1 = findIntersection(a.data(),a.size(),b.data(),b.size(),c1.data());
  });
  t.measure("Intersect SIMD",1,simdTimes,[&](){
    lenC2 = SetOp<PolicyAND>::op(a.data(),a.size(),b.data(),b.size(),c2.data());
  });

  if(lenC1 == lenC2 && std::equal(c1.data(),c1.data()+lenC1,c2.data())){
    std::cout << "Result match" << std::endl;
  } else {
    std::cout << "Result don't match" << std::endl;
    return -1;
  }

  t.measure("XOR Scalar",1,scalarTimes,[&](){
    lenC1 = findXOR(a.data(),a.size(),b.data(),b.size(),c1.data());
  });
  t.measure("XOR SIMD",1,simdTimes,[&](){
    lenC2 = SetOp<PolicyXOR>::op(a.data(),a.size(),b.data(),b.size(),c2.data());
  });

  if(lenC1 == lenC2 && std::equal(c1.data(),c1.data()+lenC1,c2.data())){
    std::cout << "Result match" << std::endl;
  } else {
    std::cout << "Result don't match" << std::endl;
    return -1;
  }

  t.measure("Diff Scalar",1,scalarTimes,[&](){
    lenC1 = findDiff(a.data(),a.size(),b.data(),b.size(),c1.data());
  });
  t.measure("Diff SIMD",1,simdTimes,[&](){
    lenC2 = SetOp<PolicyDiff>::op(a.data(),a.size(),b.data(),b.size(),c2.data());
  });

  if(lenC1 == lenC2 && std::equal(c1.data(),c1.data()+lenC1,c2.data())){
    std::cout << "Result match" << std::endl;
  } else {
    std::cout << "Result don't match" << std::endl;
    return -1;
  }


  t.dump();


}