#pragma once
#include <iterator>
#include <algorithm>
#include <numeric>
#include <cmath>
struct StatResult{
    double mean,sd;
    int n;
};

template<class IT>
StatResult statAnalysis(IT begin, IT end){
    double mean = 0, sd = 0;
    int n = end - begin;
    mean = std::accumulate(begin,end,double(0.0))/n;
    double square = std::inner_product(begin,end,begin,double(0.0));
    sd = std::sqrt((square - n * mean * mean)/n); // you should use n-1 in principle though ...
    return {mean,sd,n};
}