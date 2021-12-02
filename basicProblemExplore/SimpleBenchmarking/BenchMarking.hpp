#pragma once
#include <chrono>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <tuple>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <memory>
#include <utility>

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

struct TimmerHelper{
    using TimeType = decltype(std::chrono::steady_clock().now());
    std::vector<double> *targetVec;
    std::chrono::steady_clock clock;
    TimeType startTime;

    void start(){
        startTime = clock.now();
    }

    void end(){
        auto now = clock.now();
        auto duration = now - startTime;
        targetVec->emplace_back(duration.count());
    }

    TimmerHelper(std::vector<double> *targetVec):targetVec(targetVec){}

};

struct Timmer
{
    struct Entry{
        std::string name;
        bool isWarmup;
        std::unique_ptr<std::vector<double>> data;
    };
    std::vector<Entry> records;

    template <class Callable>
    Timmer &measure(std::string name, int warmup, int times, Callable fun)
    {
        auto timmerHelperWarmup = AllocateTimmerHelper(name+"_warmup",true);
        auto timmerHelper = AllocateTimmerHelper(name);

        for (int i = 0; i < warmup; ++i)
        {
            timmerHelperWarmup.start();
            fun();
            timmerHelperWarmup.end();
        }
        
        for (int i = 0; i < times; ++i)
        {
            timmerHelper.start();
            fun();
            timmerHelper.end();
        }
        return *this;
    }

    template <class Callable>
    Timmer &selfAssistMeasure(std::string name, int warmup, int times, Callable fun)
    {
        auto timmerHelperWarmup = AllocateTimmerHelper(name+"_warmup",true);
        auto timmerHelper = AllocateTimmerHelper(name);

        for (int i = 0; i < warmup; ++i)
        {
            fun(timmerHelperWarmup);
        }
        
        for (int i = 0; i < times; ++i)
        {
            fun(timmerHelper);
        }
        return *this;
    }

    void dump(bool detail = false, bool showWarmup = false)
    {
        for (const auto &record : records)
        {
            if (record.isWarmup && !showWarmup) continue;
            auto name = record.name.c_str();
            auto stats = statAnalysis(record.data.get()->begin(), record.data.get()->end());
            printf("[%15s]\tmean = %10.2e(ns)\tsd = %10.2e(ns)\tnsample = %5d\n", name, stats.mean, stats.sd, stats.n);
            if (detail)
            {
                for (int i = 0; i < record.data.get()->size(); ++i)
                {
                    printf("[%15s][Sample %4d]\t time = %10.2e(ns)\n", name, i, (*record.data)[i]);
                }
            }
        }
    }

    TimmerHelper AllocateTimmerHelper(std::string name, bool isWarmup=false){
        records.emplace_back(name,isWarmup, std::make_unique<std::vector<double>>());
        return TimmerHelper(records.back().data.get());
    }
};
