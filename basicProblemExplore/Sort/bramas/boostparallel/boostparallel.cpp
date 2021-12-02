/// Berenger Bramas (berenger.bramas@inria.fr)
///
/// This file implements parallel sort with the STL
/// and is used to compare with our SVE arm sort.
///
/// Need:
/// module load boost/1.73.0/arm-21.0
///
/// Compile with:
/// g++ -march=armv8.2-a+sve -DNDEBUG -O3 boostparallel.cpp -o boostparallel.cpp.exe -fopenmp

#include <vector>
#include <algorithm>


#include <iostream>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <stdexcept>
#include <climits>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <string.h>
#include <array>
#include <vector>
#include <fstream>

#include <iostream>
#include <memory>
#include <cstdlib>

#include <boost/sort/sort.hpp>


class dtimer {
    using double_second_time = std::chrono::duration<double, std::ratio<1, 1>>;

    std::chrono::high_resolution_clock::time_point
    m_start;  ///< m_start time (start)
    std::chrono::high_resolution_clock::time_point m_end;  ///< stop time (stop)
    std::chrono::nanoseconds m_cumulate;  ///< the m_cumulate time

public:
    /// Constructor
    dtimer() { start(); }

    /// Copy constructor
    dtimer(const dtimer& other) = delete;
    /// Copies an other timer
    dtimer& operator=(const dtimer& other) = delete;
    /// Move constructor
    dtimer(dtimer&& other) = delete;
    /// Copies an other timer
    dtimer& operator=(dtimer&& other) = delete;

    /** Rest all the values, and apply start */
    void reset() {
        m_start = std::chrono::high_resolution_clock::time_point();
        m_end = std::chrono::high_resolution_clock::time_point();
        m_cumulate = std::chrono::nanoseconds();
        start();
    }

    /** Start the timer */
    void start() {
        m_start = std::chrono::high_resolution_clock::now();
    }

    /** Stop the current timer */
    void stop() {
        m_end = std::chrono::high_resolution_clock::now();
        m_cumulate += std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start);
    }

    /** Return the elapsed time between start and stop (in second) */
    double getElapsed() const {
        return std::chrono::duration_cast<double_second_time>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start)).count();
    }

    /** Return the total counted time */
    double getCumulated() const {
        return std::chrono::duration_cast<double_second_time>(m_cumulate).count();
    }

    /** End the current counter (stop) and return the elapsed time */
    double stopAndGetElapsed() {
        stop();
        return getElapsed();
    }
};



template <class NumType>
void createRandVecInc(NumType array[], const int size, const NumType starValue = 0){
    if(size){
        array[0] = starValue + rand()/(RAND_MAX/5);
        for(int idx = 1 ; idx < size ; ++idx){
            array[idx] = (rand()/(RAND_MAX/5)) + array[idx-1];
        }
    }
}

template <class NumType>
void createRandVec(NumType array[], const size_t size){
    for(size_t idx = 0 ; idx < size ; ++idx){
        array[idx] = NumType(drand48()*double(size));
    }
}


// To ensure vec is used and to kill extra optimization
template <class NumType>
void useVec(NumType array[], const size_t size){
    double all = 0;
    for(size_t idx = 0 ; idx < size ; ++idx){
        all += double(array[idx]) * 0.000000000001;
    }
    // This will never happen!
    if(all == std::numeric_limits<double>::max()){
        std::cout << "The impossible happens!!" << std::endl;
        exit(99);
    }
}

template <class NumType>
void useVec(std::pair<NumType,NumType> array[], const size_t size){
    double all = 0;
    for(size_t idx = 0 ; idx < size ; ++idx){
        all += double(array[idx].first) * 0.000000000001;
    }
    // This will never happen!
    if(all == std::numeric_limits<double>::max()){
        std::cout << "The impossible happens!!" << std::endl;
        exit(99);
    }
}


std::vector<int> GetNbThreadsToTest(){
    std::vector<int> nbThreads;
    const int maxThreads = 48;
    for(int idx = 1 ; idx <= maxThreads ; idx *= 2){
        nbThreads.push_back(idx);
    }
    if(((maxThreads-1)&maxThreads) != 0){
        nbThreads.push_back(maxThreads);
    }
    return nbThreads;
}

const size_t GlobalMaxSize = 3L*1024L*1024L*1024L;

template <class NumType>
void timeAllOmp(std::ostream& fres){
    const size_t MaxSize = GlobalMaxSize;//10*1024*1024*1024;262144*8;//
    const int NbLoops = 5;
    
    const auto nbThreadsToTest = GetNbThreadsToTest();
    
    int nbSize=0;
    for(size_t currentSize = 512 ; currentSize <= MaxSize ; currentSize *= 8){
        nbSize += 1;
    }

    const bool full = false;
    const int nbAlgo = 1;
    
    std::vector<double> allTimes(3*nbAlgo*nbThreadsToTest.size()*nbSize);

    auto access = [&](int idxSize, int idxThread, int idxType, int res) -> int{
        assert(idxSize < nbSize);
        assert(idxType < nbAlgo);
        assert(res < 3);
        assert(idxThread < nbThreadsToTest.size());
        assert((((idxSize*nbThreadsToTest.size())+idxThread)*nbAlgo+idxType)*3+res < allTimes.size());
        return (((idxSize*nbThreadsToTest.size())+idxThread)*nbAlgo+idxType)*3+res;
    };
    
    for(int idxThread = 0; idxThread < nbThreadsToTest.size() ; ++idxThread){
    
        const int nbThreads = nbThreadsToTest[idxThread];
        std::cout << "nb threads " << nbThreads << std::endl;
        
        int idxSize = 0;
        for(size_t currentSize = 512 ; currentSize <= MaxSize ; currentSize *= 8 , idxSize += 1){
            std::unique_ptr<NumType[]> array(new NumType[currentSize]);
        
            std::cout << "currentSize " << currentSize << std::endl;

            for(int idxType = 0 ; idxType < nbAlgo ; ++idxType){
                allTimes[access(idxSize, idxThread, idxType, 0)] = std::numeric_limits<double>::max();
                allTimes[access(idxSize, idxThread, idxType, 1)] = std::numeric_limits<double>::min();
                allTimes[access(idxSize, idxThread, idxType, 2)] = 0;
            }

            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::cout << "  idxLoop " << idxLoop << std::endl;

                int idxType = 0;
                {
                    srand48((long int)(idxLoop));
                    createRandVec(array.get(), currentSize);
                    dtimer timer;
                    boost::sort::block_indirect_sort(&array[0], &array[currentSize],nbThreads);
                    timer.stop();
                    std::cout << "    stdsort " << timer.getElapsed() << std::endl;
                    useVec(array.get(), currentSize);
                    allTimes[access(idxSize, idxThread, idxType, 0)] = std::min(allTimes[access(idxSize, idxThread, idxType, 0)], timer.getElapsed());
                    allTimes[access(idxSize, idxThread, idxType, 1)] = std::max(allTimes[access(idxSize, idxThread, idxType, 1)], timer.getElapsed());
                    allTimes[access(idxSize, idxThread, idxType, 2)] += timer.getElapsed()/double(NbLoops);
                    idxType += 1;
                }
            }
        }
    }
    
    const char* labels[1] = {"block_indirect_sort"};
    
    fres << "#size";
    
    for(const int nbThreads : nbThreadsToTest){
        for(int idxlabel = 0 ; idxlabel < nbAlgo ; ++idxlabel){
            fres << "\t" << labels[idxlabel] << nbThreads << "min";
            fres << "\t" << labels[idxlabel] << nbThreads << "max";
            fres << "\t" << labels[idxlabel] << nbThreads << "avg";
            fres << "\t" << labels[idxlabel] << nbThreads << "avgnlogn";
            fres << "\t" << labels[idxlabel] << nbThreads << "eff";
        }
    }
    fres << "\n";
    
    int idxSize = 0;
    for(size_t currentSize = 512 ; currentSize <= MaxSize ; currentSize *= 8 , idxSize += 1){
        fres << currentSize;
        for(int idxThread = 0; idxThread < nbThreadsToTest.size() ; ++idxThread){
            for(int idxlabel = 0 ; idxlabel < nbAlgo ; ++idxlabel){
                for(int idxres = 0 ; idxres < 3 ; ++idxres){
                    fres << "\t" << allTimes[access(idxSize, idxThread, idxlabel, idxres)];
                }
                fres << "\t" << (allTimes[access(idxSize, idxThread, idxlabel, 2)]
                                /(currentSize*std::log(currentSize)));
                fres << "\t" << (allTimes[access(idxSize, 0, idxlabel, 2)])
                                /(allTimes[access(idxSize, idxThread, idxlabel, 2)]);
            }
        }
        fres << "\n";
    }    
}


int main(){    
    {
        std::ofstream fres("std-res-int-openmp.data");
        timeAllOmp<int>(fres);
    }
    {
        std::ofstream fres("std-res-double-openmp.data");
        timeAllOmp<double>(fres);
    }

    return 0;
}
