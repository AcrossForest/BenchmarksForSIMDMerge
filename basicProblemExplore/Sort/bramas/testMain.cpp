#include "sortSVE.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <memory>

int main(int argc, char** argv){
    int len = 100;
    std::vector<int> a(len);
    for(int i=0; i<len; ++i){
        a[i] = -i;
    }
    auto b = a;

    std::sort(b.begin(),b.end());

    SortSVE::Sort<int,size_t>(a.data(),len);


    std::cout << "Is result the same?" <<  (a==b) << std::endl;
    for(int i=0; i<10; ++i){
        std::cout << a[i] << '\t' << b[i] << std::endl;
    }



    
}

