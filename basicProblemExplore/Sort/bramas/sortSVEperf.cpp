//////////////////////////////////////////////////////////
/// By berenger.bramas@inria.fr 2020.
/// Licence is MIT.
/// Comes without any warranty.
///
/// Code to test the performance of the different sorts
/// and partitioning schemes.
///
/// Please refer to the README to know how to build
/// and to have more information about the functions.
///
//////////////////////////////////////////////////////////

#if defined(_OPENMP)
#include <omp.h>
#endif

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

#include "sortSVE.hpp"
#include "sortSVEkv.hpp"
#include "sortSVE512.hpp"
#include "sortSVEkv512.hpp"

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

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


////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
/// Init functions
////////////////////////////////////////////////////////////

#include <iostream>
#include <memory>
#include <cstdlib>

template <class NumType>
void assertNotSorted(const NumType array[], const size_t size, const std::string log){
    for(size_t idx = 1 ; idx < size ; ++idx){
        if(array[idx-1] > array[idx]){
            std::cout << "assertNotSorted -- Array is not sorted\n"
                         "assertNotSorted --    - at pos " << idx << "\n"
                          "assertNotSorted --    - log " << log << std::endl;
        }
    }
}

template <class NumType>
void assertNotPartitioned(const NumType array[], const size_t size, const NumType pivot,
                          const size_t limite, const std::string log){
    for(size_t idx = 0 ; idx < limite ; ++idx){
        if(array[idx] > pivot){
            std::cout << "assertNotPartitioned -- Array is not partitioned\n"
                         "assertNotPartitioned --    - at pos " << idx << "\n"
                         "assertNotPartitioned --    - log " << log << std::endl;
        }
    }
    for(size_t idx = limite ; idx < size ; ++idx){
        if(array[idx] <= pivot){
            std::cout << "assertNotPartitioned -- Array is not partitioned\n"
                         "assertNotPartitioned --    - at pos " << idx << "\n"
                         "assertNotPartitioned --    - log " << log << std::endl;
        }
    }
}

template <class NumType>
void assertNotEqual(const NumType array1[], const NumType array2[],
                    const int size, const std::string log){
    for(int idx = 0 ; idx < size ; ++idx){
        if(array1[idx] != array2[idx]){
            std::cout << "assertNotEqual -- Array is not equal\n"
                         "assertNotEqual --    - at pos " << idx << "\n"
                                                                    "assertNotEqual --    - array1 " << array1[idx] << "\n"
                                                                                                                       "assertNotEqual --    - array2 " << array2[idx] << "\n"
                                                                                                                                                                          "assertNotEqual --    - log " << log << std::endl;
        }
    }
}

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

////////////////////////////////////////////////////////////
/// Timing functions
////////////////////////////////////////////////////////////

#include <fstream>

const size_t GlobalMaxSize = 3L*1024L*1024L*1024L;

template <class NumType>
void timeAll(std::ostream& fres){
    const size_t MaxSize = GlobalMaxSize;//10*1024*1024*1024;262144*8;//
    const int NbLoops = 5;

    fres << "#size\tstdsort\tstdsortnlogn\tsortSVE\tsortSVEnlogn\tsortSVEspeed";
    if(svcntb() == 512/8){
        fres << "\tsortSVE512\tsortSVE512nlogn\tsortSVE512speed";
    }
    fres << "\n";

    for(size_t currentSize = 64 ; currentSize <= MaxSize ; currentSize *= 8 ){
        std::unique_ptr<NumType[]> array(new NumType[currentSize]);
    
        std::cout << "currentSize " << currentSize << std::endl;


        double allTimes[3][3] = {{ std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                                 { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. }};

        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            std::cout << "  idxLoop " << idxLoop << std::endl;
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                std::sort(&array[0], &array[currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
                timer.stop();
                std::cout << "    std::sort " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 0;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                SortSVE::Sort<NumType, size_t>(array.get(), currentSize);
                timer.stop();
                std::cout << "    SortSVE " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 1;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            if(svcntb() == 512/8){
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                dtimer timer;
                SortSVE512::Sort<NumType, size_t>(array.get(), currentSize);
                timer.stop();
                std::cout << "    SortSVE512 " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 2;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
        }

        std::cout << currentSize << ",\"stdsort\"," << allTimes[0][0] << "," << allTimes[0][1] << "," << allTimes[0][2] << "\n";
        std::cout << currentSize << ",\"sortSVE\"," << allTimes[1][0] << "," << allTimes[1][1] << "," << allTimes[1][2] << "\n";
        if(svcntb() == 512/8){
            std::cout << currentSize << ",\"sortSVE512\"," << allTimes[2][0] << "," << allTimes[2][1] << "," << allTimes[2][2] << "\n";
        }


        fres << currentSize << "\t"
             << allTimes[0][2] << "\t" << allTimes[0][2]/(currentSize*std::log(currentSize)) << "\t"
             << allTimes[1][2] << "\t" << allTimes[1][2]/(currentSize*std::log(currentSize)) << "\t" << allTimes[0][2]/allTimes[1][2];
        if(svcntb() == 512/8){
            fres << "\t" << allTimes[2][2] << "\t" << allTimes[2][2]/(currentSize*std::log(currentSize)) << "\t" << allTimes[0][2]/allTimes[2][2];
        }

        fres << "\n";
    }

}


template <class NumType>
void timeAll_pair(std::ostream& fres){
    const size_t MaxSize = GlobalMaxSize;//10*1024*1024*1024;262144*8;//
    const int NbLoops = 5;

    fres << "#size\tstdsort\tstdsortnlogn\tsortSVE\tsortSVEnlogn\tsortSVEspeed\tsortSVEpair\tsortSVEpairnlogn\tsortSVEpairspeed";
    if(svcntb() == 512/8){
        fres << "\tsortSVE512\tsortSVE512nlogn\tsortSVE512speed";
    }
    fres << "\n";

    for(size_t currentSize = 64 ; currentSize <= MaxSize ; currentSize *= 8 ){
    
        std::cout << "currentSize " << currentSize << std::endl;

        double allTimes[4][3] = {{ std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                                 { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                                 { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. }};

        {
            std::unique_ptr<NumType[]> array(new NumType[currentSize]);
            std::unique_ptr<NumType[]> values(new NumType[currentSize]());
            std::unique_ptr<std::pair<NumType,NumType>[]> arrayPair(new std::pair<NumType,NumType>[currentSize]());

            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::cout << "  idxLoop " << idxLoop << std::endl;
                {
                    srand48((long int)(idxLoop));
                    createRandVec(array.get(), currentSize);
                    for(size_t idxItem = 0 ; idxItem < currentSize ; ++idxItem){
                        arrayPair[idxItem].first = array[idxItem];
                    }
                    dtimer timer;
                    std::sort(&arrayPair[0], &arrayPair[currentSize], [&](const std::pair<NumType,NumType>& v1, const std::pair<NumType,NumType>& v2){
                        return v1.first < v2.first;
                    });
                    timer.stop();
                    std::cout << "    std::sort " << timer.getElapsed() << std::endl;
                    useVec(array.get(), currentSize);
                    const int idxType = 0;
                    allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                    allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                    allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
                }
                {
                    srand48((long int)(idxLoop));
                    createRandVec(array.get(), currentSize);
                    dtimer timer;
                    SortSVEkv::Sort<NumType, size_t>(array.get(), values.get(), currentSize);
                    timer.stop();
                    std::cout << "    sortSVE " << timer.getElapsed() << std::endl;
                    useVec(array.get(), currentSize);
                    const int idxType = 1;
                    allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                    allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                    allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
                }
                {
                    srand48((long int)(idxLoop));
                    createRandVec(array.get(), currentSize);
                    for(size_t idxItem = 0 ; idxItem < currentSize ; ++idxItem){
                        arrayPair[idxItem].first = array[idxItem];
                    }
                    dtimer timer;
                    SortSVEkv::Sort<std::pair<NumType,NumType>, size_t>(arrayPair.get(), currentSize);
                    timer.stop();
                    std::cout << "    sortSVE " << timer.getElapsed() << std::endl;
                    useVec(array.get(), currentSize);
                    const int idxType = 2;
                    allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                    allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                    allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
                }
                if(svcntb() == 512/8){
                    srand48((long int)(idxLoop));
                    createRandVec(array.get(), currentSize);
                    dtimer timer;
                    SortSVEkv512::Sort<NumType, size_t>(array.get(), values.get(), currentSize);
                    timer.stop();
                    std::cout << "    sortSVE512 " << timer.getElapsed() << std::endl;
                    useVec(array.get(), currentSize);
                    const int idxType = 3;
                    allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                    allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                    allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
                }
            }
        }

        std::cout << currentSize << ",\"stdsort\"," << allTimes[0][0] << "," << allTimes[0][1] << "," << allTimes[0][2] << "\n";
        std::cout << currentSize << ",\"sortSVE\"," << allTimes[1][0] << "," << allTimes[1][1] << "," << allTimes[1][2] << "\n";
        std::cout << currentSize << ",\"sortSVEpair\"," << allTimes[2][0] << "," << allTimes[2][1] << "," << allTimes[2][2] << "\n";
        if(svcntb() == 512/8){
            std::cout << currentSize << ",\"sortSVE512\"," << allTimes[3][0] << "," << allTimes[3][1] << "," << allTimes[3][2] << "\n";
        }

        fres << currentSize << "\t"
             << allTimes[0][2] << "\t" << allTimes[0][2]/(currentSize*std::log(currentSize)) << "\t"
             << allTimes[1][2] << "\t" << allTimes[1][2]/(currentSize*std::log(currentSize)) << "\t" << allTimes[0][2]/allTimes[1][2] << "\t"
             << allTimes[2][2] << "\t" << allTimes[2][2]/(currentSize*std::log(currentSize)) << "\t" << allTimes[0][2]/allTimes[2][2];
        if(svcntb() == 512/8){
            fres << "\t" << allTimes[3][2] << "\t" << allTimes[3][2]/(currentSize*std::log(currentSize)) << "\t" << allTimes[0][2]/allTimes[3][2];
        }
        fres << "\n";
    }

}

#if defined(_OPENMP)

std::vector<int> GetNbThreadsToTest(){
    std::vector<int> nbThreads;
    const int maxThreads = omp_get_max_threads();
    for(int idx = 1 ; idx <= maxThreads ; idx *= 2){
        nbThreads.push_back(idx);
    }
    if(((maxThreads-1)&maxThreads) != 0){
        nbThreads.push_back(maxThreads);
    }
    return nbThreads;
}

template <class NumType>
void timeAllOmp(std::ostream& fres, const std::string prefix){
    const size_t MaxSize = GlobalMaxSize;//10*1024*1024*1024;262144*8;//
    const int NbLoops = 5;
    
    const auto nbThreadsToTest = GetNbThreadsToTest();
    
    int nbSize=0;
    for(size_t currentSize = 512 ; currentSize <= MaxSize ; currentSize *= 8){
        nbSize += 1;
    }

    const bool full = false;
    const int nbAlgo = (svcntb() == 512/8 ? (full?2:1) : 1);
    
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
        omp_set_num_threads(nbThreads);
        
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
                    SortSVE::SortOmpPartition<NumType, size_t>(array.get(), currentSize);
                    timer.stop();
                    std::cout << "    SortOmpPartition " << timer.getElapsed() << std::endl;
                    useVec(array.get(), currentSize);
                    allTimes[access(idxSize, idxThread, idxType, 0)] = std::min(allTimes[access(idxSize, idxThread, idxType, 0)], timer.getElapsed());
                    allTimes[access(idxSize, idxThread, idxType, 1)] = std::max(allTimes[access(idxSize, idxThread, idxType, 1)], timer.getElapsed());
                    allTimes[access(idxSize, idxThread, idxType, 2)] += timer.getElapsed()/double(NbLoops);
                    idxType += 1;
                }
                if(svcntb() == 512/8){
                    if(full){
                        srand48((long int)(idxLoop));
                        createRandVec(array.get(), currentSize);
                        dtimer timer;
                        SortSVE512::SortOmpPartition<NumType, size_t>(array.get(), currentSize);
                        timer.stop();
                        std::cout << "    SortOmpPartition512 " << timer.getElapsed() << std::endl;
                        useVec(array.get(), currentSize);
                        allTimes[access(idxSize, idxThread, idxType, 0)] = std::min(allTimes[access(idxSize, idxThread, idxType, 0)], timer.getElapsed());
                        allTimes[access(idxSize, idxThread, idxType, 1)] = std::max(allTimes[access(idxSize, idxThread, idxType, 1)], timer.getElapsed());
                        allTimes[access(idxSize, idxThread, idxType, 2)] += timer.getElapsed()/double(NbLoops);
                    idxType += 1;
                    }
                }
            }
        }
    }
    
    const char* labelsfull[8] = {"SortOmpPartition","SortOmpPartition512"};
    const char* labelsnotfull[1] = {"SortOmpPartition"};
    
    const char** labels = (full ? labelsfull : labelsnotfull);
    
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

#endif

template <class NumType>
void timeSmall(std::ostream& fres){
    const size_t MaxSizeV2 = 16*(svcntb()/sizeof(NumType));
    const int NbLoops = 2000;

    std::unique_ptr<NumType[]> array(new NumType[MaxSizeV2*NbLoops]);

    double allTimes[3] = {0};

        fres << "#size\tstdsort\tstdsortlogn\tsortSVE\tsortSVElogn\tsortSVEspeed";
        if(svcntb() == 512/8){
            fres << "\tsortSVE512\tsortSVE512nlogn\tsortSVE512speed";
        }
        fres << "\n";

    for(size_t currentSize = 2 ; currentSize <= MaxSizeV2 ; currentSize++ ){
        std::cout << "currentSize " << currentSize << std::endl;
        std::cout << "    std::sort " << std::endl;
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&array[idxLoop*currentSize], &array[(idxLoop+1)*currentSize], [&](const NumType& v1, const NumType& v2){
                    return v1 < v2;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        std::cout << "    newqsSVEbitfull " << std::endl;
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                SortSVE::SmallSort16V(&array[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "    sortSVE " << timer.getElapsed() << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        if(svcntb() == 512/8){
            std::cout << "    newqsSVEbitfull512 " << std::endl;
            {
                srand48((long int)(currentSize));
                for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                    useVec(&array[idxLoop*currentSize], currentSize);
                    createRandVec(&array[idxLoop*currentSize], currentSize);
                }
            }
            {
                dtimer timer;
                for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                    SortSVE512::SmallSort16V(&array[idxLoop*currentSize], currentSize);
                }
                timer.stop();
                std::cout << "    sortSVE512 " << timer.getElapsed() << std::endl;
                const int idxType = 2;
                allTimes[idxType] = timer.getElapsed()/double(NbLoops);
            }
            {
                for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                    useVec(&array[idxLoop*currentSize], currentSize);
                }
            }
        }

        fres << currentSize << "\t" << allTimes[0] << "\t" <<
                allTimes[0]/(currentSize*std::log(currentSize)) << "\t" << allTimes[1] << "\t" <<
                allTimes[1]/(currentSize*std::log(currentSize)) << "\t" << allTimes[0]/allTimes[1];
        if(svcntb() == 512/8){
            fres << "\t" << allTimes[2] << "\t" << allTimes[2]/(currentSize*std::log(currentSize)) << "\t" << allTimes[0]/allTimes[2];
        }
        fres << "\n";
    }

}



template <class NumType>
void timeSmall_pair(std::ostream& fres){
    const size_t MaxSizeV2 = 16*(svcntb()/sizeof(NumType));
    const int NbLoops = 2000;

    std::unique_ptr<NumType[]> array(new NumType[MaxSizeV2*NbLoops]);
    std::unique_ptr<NumType[]> indexes(new NumType[MaxSizeV2*NbLoops]());

    std::unique_ptr<std::pair<NumType,NumType>[]> arrayPair(new std::pair<NumType,NumType>[MaxSizeV2*NbLoops]());


    double allTimes[4] = {0};

	fres << "#size\tstdsort\tstdsortlogn\tsortSVE\tsortSVElogn\tsortSVEspeed\tsortSVEpair\tsortSVEpairlogn\ttsortSVEpairspeed";
        if(svcntb() == 512/8){
            fres << "\tsortSVE512\tsortSVE512nlogn\tsortSVE512speed";
        }
	fres << "\n";

    for(size_t currentSize = 2 ; currentSize <= MaxSizeV2 ; currentSize++ ){
        std::cout << "currentSize " << currentSize << std::endl;
        std::cout << "    std::sort " << std::endl;
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
                for(size_t idxItem = 0 ; idxItem < currentSize ; ++idxItem){
                    arrayPair[idxLoop*currentSize+idxItem].first = array[idxLoop*currentSize+idxItem];
                }
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::sort(&arrayPair[idxLoop*currentSize], &arrayPair[(idxLoop+1)*currentSize], [&](const std::pair<NumType,NumType>& v1, const std::pair<NumType,NumType>& v2){
                    return v1.first < v2.first;
                });
            }
            timer.stop();
            std::cout << "    std::sort " << timer.getElapsed() << std::endl;
            const int idxType = 0;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        std::cout << "    sortSVE " << std::endl;
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
                createRandVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                SortSVEkv::SmallSort16V(&array[idxLoop*currentSize], &indexes[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "    sortSVE " << timer.getElapsed() << std::endl;
            const int idxType = 1;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        {
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                useVec(&array[idxLoop*currentSize], currentSize);
            }
        }
        std::cout << "    sortSVEpair " << std::endl;
        {
            srand48((long int)(currentSize));
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                createRandVec(&array[idxLoop*currentSize], currentSize);
                for(size_t idxItem = 0 ; idxItem < currentSize ; ++idxItem){
                    arrayPair[idxLoop*currentSize+idxItem].first = array[idxLoop*currentSize+idxItem];
                }
            }
        }
        {
            dtimer timer;
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                SortSVEkv::SmallSort16V(&arrayPair[idxLoop*currentSize], currentSize);
            }
            timer.stop();
            std::cout << "    sortSVEpair " << timer.getElapsed() << std::endl;
            const int idxType = 2;
            allTimes[idxType] = timer.getElapsed()/double(NbLoops);
        }
        if(svcntb() == 512/8){
            std::cout << "    sortSVE512 " << std::endl;
            {
                srand48((long int)(currentSize));
                for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                    useVec(&array[idxLoop*currentSize], currentSize);
                    createRandVec(&array[idxLoop*currentSize], currentSize);
                }
            }
            {
                dtimer timer;
                for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                    SortSVEkv512::SmallSort16V(&array[idxLoop*currentSize], &indexes[idxLoop*currentSize], currentSize);
                }
                timer.stop();
                std::cout << "    sortSVE512 " << timer.getElapsed() << std::endl;
                const int idxType = 3;
                allTimes[idxType] = timer.getElapsed()/double(NbLoops);
            }
            {
                for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                    useVec(&array[idxLoop*currentSize], currentSize);
                }
            }
        }

        fres << currentSize << "\t" << allTimes[0] << "\t" <<
                allTimes[0]/(currentSize*std::log(currentSize)) << "\t" << allTimes[1] << "\t" <<
                allTimes[1]/(currentSize*std::log(currentSize)) << "\t" << allTimes[0]/allTimes[1] << "\t" << allTimes[2] << "\t" <<
                allTimes[2]/(currentSize*std::log(currentSize)) << "\t" << allTimes[0]/allTimes[2];

        if(svcntb() == 512/8){
            fres <<"\t" << allTimes[3] << "\t" <<
                    allTimes[3]/(currentSize*std::log(currentSize)) << "\t" << allTimes[0]/allTimes[3];
        }
        fres << "\n";
    }
}


template <class NumType>
void timePartitionAll(std::ostream& fres){
    const size_t MaxSize = GlobalMaxSize;//10*1024*1024*1024;
    const int NbLoops = 20;

    fres << "#size\tstdpart\tstdpartn\tpartitionSVE\tpartitionSVEn\tpartitionSVEspeed";
    fres << "\n";

    for(size_t currentSize = 64 ; currentSize <= MaxSize ; currentSize *= 8 ){
        std::cout << "currentSize " << currentSize << std::endl;

        std::unique_ptr<NumType[]> array(new NumType[currentSize]);


        double allTimes[2][3] = {{ std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. }};

        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            std::cout << "  idxLoop " << idxLoop << std::endl;
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                const NumType pivot = array[(idxLoop*currentSize/NbLoops)];
                dtimer timer;
                std::partition(&array[0], &array[currentSize], [&](const NumType& v){
                    return v < pivot;
                });
                timer.stop();
                std::cout << "    std::partition " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 0;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
            {
                srand48((long int)(idxLoop));
                createRandVec(array.get(), currentSize);
                const NumType pivot = array[(idxLoop*currentSize/NbLoops)];
                dtimer timer;
                SortSVE::PartitionSVE<size_t>(array.get(), 0, currentSize-1, pivot);
                timer.stop();
                std::cout << "    partitionSVE " << timer.getElapsed() << std::endl;
                useVec(array.get(), currentSize);
                const int idxType = 1;
                allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
            }
        }

        std::cout << currentSize << ",\"stdpartion\"," << allTimes[0][0] << "," << allTimes[0][1] << "," << allTimes[0][2] << "\n";
        std::cout << currentSize << ",\"partitionSVE\"," << allTimes[1][0] << "," << allTimes[1][1] << "," << allTimes[1][2] << "\n";

        fres << currentSize << "\t"
             << allTimes[0][2] << "\t" << allTimes[0][2]/(currentSize) << "\t"
             << allTimes[1][2] << "\t" << allTimes[1][2]/(currentSize) << "\t" << allTimes[0][2]/allTimes[1][2] << "\n";
    }
}

template <class NumType>
void timePartitionAll_pair(std::ostream& fres){
    const size_t MaxSize = GlobalMaxSize;//10*1024*1024*1024;
    const int NbLoops = 20;

    fres << "#size\tstdpart\tstdpartn\tpartitionSVE\tpartitionSVEn\tpartitionSVEspeed\tpartitionSVEpair\tpartitionSVEpairn\tpartitionSVEpairspeed";    
    if(svcntb() == 512/8){
        fres << "\tpartitionSVE512\tpartitionSVEn512\tpartitionSVEspeed512";
    }
    fres << "\n";

    for(size_t currentSize = 64 ; currentSize <= MaxSize ; currentSize *= 8 ){
        std::cout << "currentSize " << currentSize << std::endl;

        double allTimes[4][3] = {{ std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                            { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. },
                                 { std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 0. }};
        {
            std::unique_ptr<NumType[]> array(new NumType[currentSize]);
            std::unique_ptr<NumType[]> values(new NumType[currentSize]());
            std::unique_ptr<std::pair<NumType,NumType>[]> arrayPair(new std::pair<NumType,NumType>[currentSize]());
            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                std::cout << "  idxLoop " << idxLoop << std::endl;
                {
                    srand48((long int)(idxLoop));
                    createRandVec(array.get(), currentSize);
                    for(size_t idxItem = 0 ; idxItem < currentSize ; ++idxItem){
                        arrayPair[idxItem].first = array[idxItem];
                    }
                    const NumType pivot = array[(idxLoop*currentSize/NbLoops)];
                    dtimer timer;
                    std::partition(&arrayPair[0], &arrayPair[currentSize], [&](const std::pair<NumType,NumType>& v){
                        return v.first < pivot;
                    });
                    timer.stop();
                    std::cout << "    std::partition " << timer.getElapsed() << std::endl;
                    useVec(array.get(), currentSize);
                    const int idxType = 0;
                    allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                    allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                    allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
                }
                {
                    srand48((long int)(idxLoop));
                    createRandVec(array.get(), currentSize);
                    const NumType pivot = array[(idxLoop*currentSize/NbLoops)];
                    dtimer timer;
                    SortSVEkv::PartitionSVE<size_t>(array.get(), values.get(), 0, currentSize-1, pivot);
                    timer.stop();
                    std::cout << "    partitionSVE " << timer.getElapsed() << std::endl;
                    useVec(array.get(), currentSize);
                    const int idxType = 1;
                    allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                    allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                    allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
                }                          
                {
                    srand48((long int)(idxLoop));
                    createRandVec(array.get(), currentSize);
                    for(size_t idxItem = 0 ; idxItem < currentSize ; ++idxItem){
                        arrayPair[idxItem].first = array[idxItem];
                    }
                    const NumType pivot = array[(idxLoop*currentSize/NbLoops)];
                    dtimer timer;
                    SortSVEkv::PartitionSVE<size_t>(arrayPair.get(), 0, currentSize-1, pivot);
                    timer.stop();
                    std::cout << "    partitionSVE " << timer.getElapsed() << std::endl;
                    useVec(array.get(), currentSize);
                    const int idxType = 2;
                    allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                    allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                    allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
                }
                if(svcntb() == 512/8){
                    srand48((long int)(idxLoop));
                    createRandVec(array.get(), currentSize);
                    const NumType pivot = array[(idxLoop*currentSize/NbLoops)];
                    dtimer timer;
                    SortSVEkv512::PartitionSVE<size_t>(array.get(), values.get(), 0, currentSize-1, pivot);
                    timer.stop();
                    std::cout << "    partitionSVE512 " << timer.getElapsed() << std::endl;
                    useVec(array.get(), currentSize);
                    const int idxType = 3;
                    allTimes[idxType][0] = std::min(allTimes[idxType][0], timer.getElapsed());
                    allTimes[idxType][1] = std::max(allTimes[idxType][1], timer.getElapsed());
                    allTimes[idxType][2] += timer.getElapsed()/double(NbLoops);
                }
            }
        }

        std::cout << currentSize << ",\"stdpartion\"," << allTimes[0][0] << "," << allTimes[0][1] << "," << allTimes[0][2] << "\n";
        std::cout << currentSize << ",\"partitionSVEV2\"," << allTimes[1][0] << "," << allTimes[1][1] << "," << allTimes[1][2] << "\n";
        std::cout << currentSize << ",\"partitionSVEV2pair\"," << allTimes[1][0] << "," << allTimes[1][1] << "," << allTimes[1][2] << "\n";
        if(svcntb() == 512/8){
            std::cout << currentSize << ",\"partitionSVEV2512\"," << allTimes[3][0] << "," << allTimes[3][1] << "," << allTimes[3][2] << "\n";
        }

        fres << currentSize << "\t"
             << allTimes[0][2] << "\t" << allTimes[0][2]/(currentSize) << "\t"
             << allTimes[1][2] << "\t" << allTimes[1][2]/(currentSize) << "\t" << allTimes[0][2]/allTimes[1][2] << "\t"
             << allTimes[2][2] << "\t" << allTimes[2][2]/(currentSize) << "\t" << allTimes[0][2]/allTimes[2][2];
        if(svcntb() == 512/8){
            fres << "\t" << allTimes[3][2] << "\t" << allTimes[3][2]/(currentSize) << "\t" << allTimes[0][2]/allTimes[3][2];
        }
        fres << "\n";
    }
}


////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

int main(int argc, char** argv){
    if(argc == 2 && strcmp(argv[1], "seq") == 0){
        {
            std::ofstream fres("smallres-int.data");
            timeSmall<int>(fres);
        }
        {
            std::ofstream fres("smallres-double.data");
            timeSmall<double>(fres);
        }
        {
            std::ofstream fres("smallres-pair-int.data");
            timeSmall_pair<int>(fres);
        }
        {
            std::ofstream fres("partitions-int.data");
            timePartitionAll<int>(fres);
        }
        {
            std::ofstream fres("partitions-double.data");
            timePartitionAll<double>(fres);
        }
        {
            std::ofstream fres("partitions-pair-int.data");
            timePartitionAll_pair<int>(fres);
        }
        {
            std::ofstream fres("res-int.data");
            timeAll<int>(fres);
        }
        {
            std::ofstream fres("res-double.data");
            timeAll<double>(fres);
        }
        {
            std::ofstream fres("res-pair-int.data");
            timeAll_pair<int>(fres);
        }
    }
#if defined(_OPENMP)
    else if(argc == 2 && strcmp(argv[1], "par") == 0){
        {
            std::ofstream fres("res-int-openmp.data");
            timeAllOmp<int>(fres, "max-threads");
        }
        {
            std::ofstream fres("res-double-openmp.data");
            timeAllOmp<double>(fres, "max-threads");
        }
    }
    else{
        std::cout << "Command should be: " << argv[0] << "[seq|par]\n";
    }
#else
    else{
        std::cout << "OpenMP not found, command should be: " << argv[0] << "[seq]\n";
    }
#endif
    return 0;
}
