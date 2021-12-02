//////////////////////////////////////////////////////////
/// By berenger.bramas@inria.fr 2020.
/// Licence is MIT.
/// Comes without any warranty.
///
/// Code to test the correctness of the different sorts
/// and partitioning schemes.
///
/// Please refer to the README to know how to build
/// and to have more information about the functions.
///
//////////////////////////////////////////////////////////

#include "sortSVE.hpp"
#include "sortSVE512.hpp"
#include "sortSVEkv.hpp"
#include "sortSVEkv512.hpp"

#include <iostream>
#include <memory>
#include <cstdlib>
#include <functional>
#include <limits>

int test_res = 0;


template <class NumType>
void assertNotSorted(const NumType array[], const size_t size, const std::string log){
    for(size_t idx = 1 ; idx < size ; ++idx){
        if(array[idx-1] > array[idx]){
            std::cout << "assertNotSorted -- Array is not sorted\n"
                         "assertNotSorted --    - at pos " << idx << "\n"
                                                                     "assertNotSorted --    - log " << log << std::endl;
            test_res = 1;
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
            test_res = 1;
        }
    }
    for(size_t idx = limite ; idx < size ; ++idx){
        if(array[idx] <= pivot){
            std::cout << "assertNotPartitioned -- Array is not partitioned\n"
                         "assertNotPartitioned --    - at pos " << idx << "\n"
                                                                          "assertNotPartitioned --    - log " << log << std::endl;
            test_res = 1;
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
            test_res = 1;
        }
    }
}

template <class NumType>
void createRandVec(NumType array[], const size_t size){
    for(size_t idx = 0 ; idx < size ; ++idx){
        array[idx] = NumType(drand48()*double(size));
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

#include <cstring>

template <class NumType, class SizeType = size_t>
class Checker{
    std::unique_ptr<NumType[]> cpArray;
    NumType* ptrArray;
    SizeType size;
public:
    Checker(const NumType sourceArray[],
            NumType toCheck[],
            const SizeType inSinze)
        : ptrArray(toCheck), size(inSinze){
        cpArray.reset(new NumType[size]);
        memcpy(cpArray.get(), sourceArray, size*sizeof(NumType));
    }

    ~Checker(){
        std::sort(ptrArray, ptrArray+size);
        std::sort(cpArray.get(), cpArray.get()+size);
        assertNotEqual(cpArray.get(), ptrArray, size, "Checker");
    }
};


void testSortVec_Core_Equal(const double toSort[], const double sorted[]){
    {
        double res[svcntb()/sizeof(double)];

        svfloat64_t vec = svld1_f64(svptrue_b64(), toSort);
        SortSVE::CoreSmallSort(vec);
        svst1_f64(svptrue_b64(), res, vec);
        assertNotSorted(res, svcntb()/sizeof(double), "testSortVec_Core_Equal");
        assertNotEqual(res, sorted, svcntb()/sizeof(double), "testSortVec_Core_Equal");
    }
    if(svcntb() == 512/8){
        double res[svcntb()/sizeof(double)];

        svfloat64_t vec = svld1_f64(svptrue_b64(), toSort);
        SortSVE512::CoreSmallSort(vec);
        svst1_f64(svptrue_b64(), res, vec);
        assertNotSorted(res, svcntb()/sizeof(double), "testSortVec_Core_Equal 512");
        assertNotEqual(res, sorted, svcntb()/sizeof(double), "testSortVec_Core_Equal 512");
    }
}


void testSortVec_Core_Equal(const int toSort[], const int sorted[]){
    {
        int res[svcntb()/sizeof(int)];

        svint32_t vec = svld1_s32(svptrue_b32(), toSort);
        SortSVE::CoreSmallSort(vec);
        svst1_s32(svptrue_b32(), res, vec);
        assertNotSorted(res, svcntb()/sizeof(int), "testSortVec_Core_Equal");
        assertNotEqual(res, sorted, svcntb()/sizeof(int), "testSortVec_Core_Equal");
    }
    if(svcntb() == 512/8){
        int res[svcntb()/sizeof(int)];

        svint32_t vec = svld1_s32(svptrue_b32(), toSort);
        SortSVE512::CoreSmallSort(vec);
        svst1_s32(svptrue_b32(), res, vec);
        assertNotSorted(res, svcntb()/sizeof(int), "testSortVec_Core_Equal 512");
        assertNotEqual(res, sorted, svcntb()/sizeof(int), "testSortVec_Core_Equal 512");
    }
}

void testSortVec(){
    std::cout << "Start testSortVec double...\n";
    {
        const int SizeVec = svcntb()/sizeof(double);
        {
            double vecTest[SizeVec];
            double vecRes[SizeVec];
            for(int idx = 0 ; idx < SizeVec ; ++idx){
                vecTest[idx] = idx+1;
                vecRes[idx] = vecTest[idx];
            }
            testSortVec_Core_Equal(vecTest, vecRes);
        }
        {
            double vecTest[SizeVec];
            double vecRes[SizeVec];
            for(int idx = 0 ; idx < SizeVec ; ++idx){
                vecTest[idx] = SizeVec-idx;
            }
            for(int idx = 0 ; idx < SizeVec ; ++idx){
                vecRes[idx] = vecTest[SizeVec-idx-1];
            }
            testSortVec_Core_Equal(vecTest, vecRes);
        }
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[SizeVec];
            createRandVec(vecTest, SizeVec);

            double res[SizeVec];
            {
                Checker<double> checker(vecTest, res, SizeVec);

                svfloat64_t vec = svld1_f64(svptrue_b64(), vecTest);
                SortSVE::CoreSmallSort(vec);
                svst1_f64(svptrue_b64(), res, vec);

                assertNotSorted(res, SizeVec, "testSortVec_Core_Equal");
            }
        }        
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                double vecTest[SizeVec];
                createRandVec(vecTest, SizeVec);

                double res[SizeVec];
                {
                    Checker<double> checker(vecTest, res, SizeVec);

                    svfloat64_t vec = svld1_f64(svptrue_b64(), vecTest);
                    SortSVE512::CoreSmallSort(vec);
                    svst1_f64(svptrue_b64(), res, vec);

                    assertNotSorted(res, SizeVec, "testSortVec_Core_Equal512");
                }
            }

        }
    }
    std::cout << "Start testSortVec int...\n";
    {
        const int SizeVec = svcntb()/sizeof(int);
        {
            int vecTest[SizeVec];
            int vecRes[SizeVec];
            for(int idx = 0 ; idx < SizeVec ; ++idx){
                vecTest[idx] = idx+1;
                vecRes[idx] = vecTest[idx];
            }
            testSortVec_Core_Equal(vecTest, vecRes);
        }
        {
            int vecTest[SizeVec];
            int vecRes[SizeVec];
            for(int idx = 0 ; idx < SizeVec ; ++idx){
                vecTest[idx] = SizeVec-idx;
            }
            for(int idx = 0 ; idx < SizeVec ; ++idx){
                vecRes[idx] = vecTest[SizeVec-idx-1];
            }
            testSortVec_Core_Equal(vecTest, vecRes);
        }
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[SizeVec];
            createRandVec(vecTest, SizeVec);

            int res[SizeVec];
            {
                Checker<int> checker(vecTest, res, SizeVec);

                svint32_t vec = svld1_s32(svptrue_b32(), vecTest);
                SortSVE::CoreSmallSort(vec);
                svst1_s32(svptrue_b32(), res, vec);

                assertNotSorted(res, SizeVec, "testSortVec_Core_Equal");
            }

        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[SizeVec];
                createRandVec(vecTest, SizeVec);

                int res[SizeVec];
                {
                    Checker<int> checker(vecTest, res, SizeVec);

                    svint32_t vec = svld1_s32(svptrue_b32(), vecTest);
                    SortSVE512::CoreSmallSort(vec);
                    svst1_s32(svptrue_b32(), res, vec);

                    assertNotSorted(res, SizeVec, "testSortVec_Core_Equal512");
                }

            }

        }
    }
}

void testSortVec_pair(){
    std::cout << "Start testSortVec_pair int...\n";
    {
        const int SizeVec = svcntb()/sizeof(int);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[SizeVec];
            createRandVec(vecTest, SizeVec);

            int values[SizeVec];
            for(int idxval = 0 ; idxval < SizeVec ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }

            {
                SortSVEkv::CoreSmallSort(vecTest, values);
                assertNotSorted(vecTest, SizeVec, "testSortVec_Core_Equal");
            }
            for(int idxval = 0 ; idxval < SizeVec ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
            }

        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[SizeVec];
                createRandVec(vecTest, SizeVec);

                int values[SizeVec];
                for(int idxval = 0 ; idxval < SizeVec ; ++idxval){
                    values[idxval] = vecTest[idxval]*100+1;
                }

                {
                    SortSVEkv512::CoreSmallSort(vecTest, values);
                    assertNotSorted(vecTest, SizeVec, "testSortVec_Core_Equal512");
                }
                for(int idxval = 0 ; idxval < SizeVec ; ++idxval){
                    if(values[idxval] != vecTest[idxval]*100+1){
                        std::cout << "Error in testSortVec_pair512 "
                                     " is " << values[idxval] <<
                                     " should be " << vecTest[idxval]*100+1 << std::endl;
                        test_res = 1;
                    }
                }

            }

        }
    }
    std::cout << "Start testSortVec_pair pair int...\n";
    {
        const int SizeVec = svcntb()/sizeof(int);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[SizeVec];
            createRandVec(vecTest, SizeVec);

            int values[SizeVec];
            for(int idxval = 0 ; idxval < SizeVec ; ++idxval){
                values[idxval] = vecTest[idxval]*100+1;
            }

            {
                std::pair<int,int> vecTestPair[SizeVec];
                for(int idxval = 0 ; idxval < SizeVec ; ++idxval){
                    vecTestPair[idxval].first = vecTest[idxval];
                    vecTestPair[idxval].second = values[idxval];
                }

                SortSVEkv::CoreSmallSort(vecTestPair);

                for(int idxval = 0 ; idxval < SizeVec ; ++idxval){
                    vecTest[idxval] = vecTestPair[idxval].first;
                    values[idxval] = vecTestPair[idxval].second;
                }

                assertNotSorted(vecTest, SizeVec, "testSortVec_Core_Equal");
            }
            for(int idxval = 0 ; idxval < SizeVec ; ++idxval){
                if(values[idxval] != vecTest[idxval]*100+1){
                    std::cout << "Error in testSortVec_pair "
                                 " is " << values[idxval] <<
                                 " should be " << vecTest[idxval]*100+1 << std::endl;
                    test_res = 1;
                }
            }

        }
    }
}

void testSort2Vec_Core_Equal(const double toSort[], const double sorted[]){
    {
        const int SizeVec = svcntb()/sizeof(double);
        double res[SizeVec*2];

        svfloat64_t vec1 = svld1_f64(svptrue_b64(), toSort);
        svfloat64_t vec2 = svld1_f64(svptrue_b64(), toSort+SizeVec);
        SortSVE::CoreSmallSort2(vec1, vec2);
        svst1_f64(svptrue_b64(), res, vec1);
        svst1_f64(svptrue_b64(), res+SizeVec, vec2);
        assertNotSorted(res, SizeVec*2, "testSort2Vec_Core_Equal");
        assertNotEqual(res, sorted, SizeVec*2, "testSort2Vec_Core_Equal");
    }
    if(svcntb() == 512/8){
        const int SizeVec = svcntb()/sizeof(double);
        double res[SizeVec*2];

        svfloat64_t vec1 = svld1_f64(svptrue_b64(), toSort);
        svfloat64_t vec2 = svld1_f64(svptrue_b64(), toSort+SizeVec);
        SortSVE512::CoreSmallSort2(vec1, vec2);
        svst1_f64(svptrue_b64(), res, vec1);
        svst1_f64(svptrue_b64(), res+SizeVec, vec2);
        assertNotSorted(res, SizeVec*2, "testSort2Vec_Core_Equal512");
        assertNotEqual(res, sorted, SizeVec*2, "testSort2Vec_Core_Equal512");

    }
}

void testSort2Vec_Core_Equal(const int toSort[], const int sorted[]){
    {
        const int SizeVec = svcntb()/sizeof(int);
        int res[SizeVec*2];

        svint32_t vec1 = svld1_s32(svptrue_b32(), toSort);
        svint32_t vec2 = svld1_s32(svptrue_b32(), toSort+SizeVec);
        SortSVE::CoreSmallSort2(vec1, vec2);
        svst1_s32(svptrue_b32(), res, vec1);
        svst1_s32(svptrue_b32(), res+SizeVec, vec2);
        assertNotSorted(res, SizeVec*2, "testSort2Vec_Core_Equal");
        assertNotEqual(res, sorted, SizeVec*2, "testSort2Vec_Core_Equal");
    }
    if(svcntb() == 512/8){
        const int SizeVec = svcntb()/sizeof(int);
        int res[SizeVec*2];

        svint32_t vec1 = svld1_s32(svptrue_b32(), toSort);
        svint32_t vec2 = svld1_s32(svptrue_b32(), toSort+SizeVec);
        SortSVE512::CoreSmallSort2(vec1, vec2);
        svst1_s32(svptrue_b32(), res, vec1);
        svst1_s32(svptrue_b32(), res+SizeVec, vec2);
        assertNotSorted(res, SizeVec*2, "testSort2Vec_Core_Equal512");
        assertNotEqual(res, sorted, SizeVec*2, "testSort2Vec_Core_Equal512");

    }
}


void testSort2Vec(){
    std::cout << "Start SortSVE::CoreSmallSort2 double...\n";
    {        
        const int SizeVec = svcntb()/sizeof(double);
        {
            double vecTest[SizeVec*2];
            double vecRes[SizeVec*2];
            for(int idx = 0 ; idx < SizeVec*2 ; idx+=2){
                vecTest[idx] = idx+1;
                vecTest[idx+1] = idx+1;
                vecRes[idx] = vecTest[idx];
                vecRes[idx+1] = vecTest[idx+1];
            }
            testSort2Vec_Core_Equal(vecTest, vecRes);
        }
        {
            double vecTest[SizeVec*2];
            double vecRes[SizeVec*2];
            for(int idx = 0 ; idx < SizeVec*2 ; idx+=2){
                vecTest[idx] = SizeVec-idx;
                vecTest[idx+1] = SizeVec-idx;
            }

            for(int idx = 0 ; idx < SizeVec*2 ; idx+=2){
                vecRes[idx] = vecTest[(SizeVec*2)-idx-2];
                vecRes[idx+1] = vecTest[(SizeVec*2)-idx-1];
            }
            testSort2Vec_Core_Equal(vecTest, vecRes);
        }
        {
            double vecTest[SizeVec*2];
            double vecRes[SizeVec*2];
            for(int idx = 0 ; idx < SizeVec*2 ; idx+=2){
                vecRes[idx] = idx+1;
                vecRes[idx+1] = idx+1;
            }

            for(int idx = 0 ; idx < SizeVec*2 ; idx+=1){
                vecTest[idx] = vecRes[(idx+4)%(SizeVec*2)];
            }

            testSort2Vec_Core_Equal(vecTest, vecRes);
        }
        {
            double vecTest[SizeVec*2];
            double vecRes[SizeVec*2];
            for(int idx = 0 ; idx < SizeVec*2 ; idx+=2){
                vecRes[idx] = idx+1;
                vecRes[idx+1] = idx+1;
            }

            for(int idx = 0 ; idx < SizeVec*2 ; idx+=1){
                vecTest[idx] = vecRes[(idx+8)%(SizeVec*2)];
            }

            testSort2Vec_Core_Equal(vecTest, vecRes);
        }
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[SizeVec*2];
            createRandVec(vecTest, SizeVec*2);

            {
                createRandVec(vecTest, SizeVec*2);
                Checker<double> checker(vecTest, vecTest, SizeVec*2);
                SortSVE::CoreSmallSort2(vecTest, vecTest+SizeVec);
                assertNotSorted(vecTest, SizeVec*2, "testSortVec_Core_Equal");
            }
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                double vecTest[SizeVec*2];
                createRandVec(vecTest, SizeVec*2);

                {
                    createRandVec(vecTest, SizeVec*2);
                    Checker<double> checker(vecTest, vecTest, SizeVec*2);
                    SortSVE512::CoreSmallSort2(vecTest, vecTest+SizeVec);
                    assertNotSorted(vecTest, SizeVec*2, "testSortVec_Core_Equal512");
                }
            }

        }
    }
    std::cout << "Start testSort2Vec int...\n";
    {
        const int SizeVec = svcntb()/sizeof(int);
        {
            int vecTest[SizeVec*2];
            int vecRes[SizeVec*2];
            for(int idx = 0 ; idx < SizeVec*2 ; idx+=2){
                vecTest[idx] = idx+1;
                vecTest[idx+1] = idx+1;
                vecRes[idx] = vecTest[idx];
                vecRes[idx+1] = vecTest[idx+1];
            }
            testSort2Vec_Core_Equal(vecTest, vecRes);
        }
        {
            int vecTest[SizeVec*2];
            int vecRes[SizeVec*2];
            for(int idx = 0 ; idx < SizeVec*2 ; idx+=2){
                vecTest[idx] = SizeVec-idx;
                vecTest[idx+1] = SizeVec-idx;
            }

            for(int idx = 0 ; idx < SizeVec*2 ; idx+=2){
                vecRes[idx] = vecTest[(SizeVec*2)-idx-2];
                vecRes[idx+1] = vecTest[(SizeVec*2)-idx-1];
            }
            testSort2Vec_Core_Equal(vecTest, vecRes);
        }
        {
            int vecTest[SizeVec*2];
            int vecRes[SizeVec*2];
            for(int idx = 0 ; idx < SizeVec*2 ; idx+=2){
                vecRes[idx] = idx+1;
                vecRes[idx+1] = idx+1;
            }

            for(int idx = 0 ; idx < SizeVec*2 ; idx+=1){
                vecTest[idx] = vecRes[(idx+4)%(SizeVec*2)];
            }

            testSort2Vec_Core_Equal(vecTest, vecRes);
        }
        {
            int vecTest[SizeVec*2];
            int vecRes[SizeVec*2];
            for(int idx = 0 ; idx < SizeVec*2 ; idx+=1){
                vecRes[idx] = idx+1;
                vecRes[idx+1] = idx+1;
            }

            for(int idx = 0 ; idx < SizeVec*2 ; idx++){
                vecTest[idx] = vecRes[(idx+8)%(SizeVec*2)];
            }

            testSort2Vec_Core_Equal(vecTest, vecRes);
        }
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[SizeVec*2];
            createRandVec(vecTest, SizeVec*2);

            {
                createRandVec(vecTest, SizeVec*2);
                Checker<int> checker(vecTest, vecTest, SizeVec*2);
                SortSVE::CoreSmallSort2(vecTest, vecTest+SizeVec);
                assertNotSorted(vecTest, SizeVec*2, "testSortVec_Core_Equal");
            }
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[SizeVec*2];
                createRandVec(vecTest, SizeVec*2);

                {
                    createRandVec(vecTest, SizeVec*2);
                    Checker<int> checker(vecTest, vecTest, SizeVec*2);
                    SortSVE512::CoreSmallSort2(vecTest, vecTest+SizeVec);
                    assertNotSorted(vecTest, SizeVec*2, "testSortVec_Core_Equal512");
                }
            }

        }
    }
}

void testSort3Vec(){
    const int nbVecs = 3;
    std::cout << "Start testSort3Vec double...\n";
    {
        const int sizeVec = svcntb()/sizeof(double);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort3(vecTest, vecTest+sizeVec, vecTest+sizeVec*2);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                double vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort3(vecTest, vecTest+sizeVec, vecTest+sizeVec*2);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
    std::cout << "Start testSort3Vec int...\n";
    {
        const int sizeVec = svcntb()/sizeof(int);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort3(vecTest, vecTest+sizeVec, vecTest+sizeVec*2);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort3(vecTest, vecTest+sizeVec, vecTest+sizeVec*2);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
}

void testSort4Vec(){
    const int nbVecs = 4;
    std::cout << "Start testSort4Vec double...\n";
    {
        const int sizeVec = svcntb()/sizeof(double);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort4(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                double vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort4(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
    std::cout << "Start testSort4Vec int...\n";
    {
        const int sizeVec = svcntb()/sizeof(int);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort4(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort4(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
}

void testSort5Vec(){
    const int nbVecs = 5;
    std::cout << "Start testSort5Vec double...\n";
    {
        const int sizeVec = svcntb()/sizeof(double);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort5(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                double vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort5(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
    std::cout << "Start testSort5Vec int...\n";
    {
        const int sizeVec = svcntb()/sizeof(int);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort5(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort5(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
}

void testSort6Vec(){
    const int nbVecs = 6;
    std::cout << "Start testSort6Vec double...\n";
    {
        const int sizeVec = svcntb()/sizeof(double);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort6(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                double vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort6(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
    std::cout << "Start testSort6Vec int...\n";
    {
        const int sizeVec = svcntb()/sizeof(int);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort6(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort6(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
}

void testSort7Vec(){
    const int nbVecs = 7;
    std::cout << "Start testSort7Vec double...\n";
    {
        const int sizeVec = svcntb()/sizeof(double);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort7(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                double vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort7(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
    std::cout << "Start testSort7Vec int...\n";
    {
        const int sizeVec = svcntb()/sizeof(int);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort7(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort7(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
}

void testSort8Vec(){
    const int nbVecs = 8;
    std::cout << "Start testSort8Vec double...\n";
    {
        const int sizeVec = svcntb()/sizeof(double);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort8(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                double vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort8(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
    std::cout << "Start testSort8Vec int...\n";
    {
        const int sizeVec = svcntb()/sizeof(int);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort8(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort8(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
}


void testSort9Vec(){
    const int nbVecs = 9;
    std::cout << "Start testSort9Vec double...\n";
    {
        const int sizeVec = svcntb()/sizeof(double);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort9(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                    vecTest+sizeVec*8);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                double vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort9(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                        vecTest+sizeVec*8);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
    std::cout << "Start testSort9Vec int...\n";
    {
        const int sizeVec = svcntb()/sizeof(int);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort9(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                    vecTest+sizeVec*8);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort9(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                        vecTest+sizeVec*8);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
}

void testSort10Vec(){
    const int nbVecs = 10;
    std::cout << "Start testSort10Vec double...\n";
    {
        const int sizeVec = svcntb()/sizeof(double);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort10(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                double vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort10(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                         vecTest+sizeVec*8, vecTest+sizeVec*9);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
    std::cout << "Start testSort10Vec int...\n";
    {
        const int sizeVec = svcntb()/sizeof(int);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort10(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort10(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                         vecTest+sizeVec*8, vecTest+sizeVec*9);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
}


void testSort11Vec(){
    const int nbVecs = 11;
    std::cout << "Start testSort11Vec double...\n";
    {
        const int sizeVec = svcntb()/sizeof(double);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort11(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                double vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort11(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                         vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
    std::cout << "Start testSort11Vec int...\n";
    {
        const int sizeVec = svcntb()/sizeof(int);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort11(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort11(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                         vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
}

void testSort12Vec(){
    const int nbVecs = 12;
    std::cout << "Start testSort12Vec double...\n";
    {
        const int sizeVec = svcntb()/sizeof(double);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort12(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                double vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort12(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                         vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
    std::cout << "Start testSort12Vec int...\n";
    {
        const int sizeVec = svcntb()/sizeof(int);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort12(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort12(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                         vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
}

void testSort13Vec(){
    const int nbVecs = 13;
    std::cout << "Start testSort13Vec double...\n";
    {
        const int sizeVec = svcntb()/sizeof(double);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort13(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                double vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort13(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                         vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
    std::cout << "Start testSort13Vec int...\n";
    {
        const int sizeVec = svcntb()/sizeof(int);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort13(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort13(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                         vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
}

void testSort14Vec(){
    const int nbVecs = 14;
    std::cout << "Start testSort14Vec double...\n";
    {
        const int sizeVec = svcntb()/sizeof(double);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort14(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*sizeVec, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                double vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort14(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                         vecTest+sizeVec*sizeVec, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
    std::cout << "Start testSort14Vec int...\n";
    {
        const int sizeVec = svcntb()/sizeof(int);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort14(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort14(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                         vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
}

void testSort15Vec(){
    const int nbVecs = 15;
    std::cout << "Start testSort15Vec double...\n";
    {
        const int sizeVec = svcntb()/sizeof(double);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort15(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*sizeVec, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13, vecTest+sizeVec*14);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                double vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort15(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                         vecTest+sizeVec*sizeVec, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13, vecTest+sizeVec*14);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
    std::cout << "Start testSort15Vec int...\n";
    {
        const int sizeVec = svcntb()/sizeof(int);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort15(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13, vecTest+sizeVec*14);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort15(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                         vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13, vecTest+sizeVec*14);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
}


void testSort16Vec(){
    const int nbVecs = 16;
    std::cout << "Start testSort16Vec double...\n";
    {
        const int sizeVec = svcntb()/sizeof(double);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            double vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort16(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*sizeVec, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13, vecTest+sizeVec*14, vecTest+sizeVec*15);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                double vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<double> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort16(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                         vecTest+sizeVec*sizeVec, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13, vecTest+sizeVec*14, vecTest+sizeVec*15);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
    std::cout << "Start testSort16Vec int...\n";
    {
        const int sizeVec = svcntb()/sizeof(int);
        srand48(0);
        const static int NbLoops = 1000;
        for(int idx = 0 ; idx < NbLoops ; ++idx){
            int vecTest[nbVecs*sizeVec];

            createRandVec(vecTest, nbVecs*sizeVec);

            Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
            SortSVE::CoreSmallSort16(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                     vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13, vecTest+sizeVec*14, vecTest+sizeVec*15);
            assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal");
        }
        if(svcntb() == 512/8){
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[nbVecs*sizeVec];

                createRandVec(vecTest, nbVecs*sizeVec);

                Checker<int> checker(vecTest, vecTest, nbVecs*sizeVec);
                SortSVE512::CoreSmallSort16(vecTest, vecTest+sizeVec, vecTest+sizeVec*2, vecTest+sizeVec*3, vecTest+sizeVec*4, vecTest+sizeVec*5, vecTest+sizeVec*6, vecTest+sizeVec*7,
                                         vecTest+sizeVec*8, vecTest+sizeVec*9, vecTest+sizeVec*10, vecTest+sizeVec*11,vecTest+sizeVec*12, vecTest+sizeVec*13, vecTest+sizeVec*14, vecTest+sizeVec*15);
                assertNotSorted(vecTest, nbVecs*sizeVec, "testSortVec_Core_Equal512");
            }

        }
    }
}

void testSortXVec_pair(){
    {
        typedef void (*func_ptr)(int*, int* );
        std::function<void(int*,int*)> functions[]
                        = { (func_ptr)SortSVEkv::CoreSmallSort,
                            (func_ptr)SortSVEkv::CoreSmallSort2,
                            (func_ptr)SortSVEkv::CoreSmallSort3,
                            (func_ptr)SortSVEkv::CoreSmallSort4,
                            (func_ptr)SortSVEkv::CoreSmallSort5,
                            (func_ptr)SortSVEkv::CoreSmallSort6,
                            (func_ptr)SortSVEkv::CoreSmallSort7,
                            (func_ptr)SortSVEkv::CoreSmallSort8,
                            (func_ptr)SortSVEkv::CoreSmallSort9,
                            (func_ptr)SortSVEkv::CoreSmallSort10,
                            (func_ptr)SortSVEkv::CoreSmallSort11,
                            (func_ptr)SortSVEkv::CoreSmallSort12,
                            (func_ptr)SortSVEkv::CoreSmallSort13,
                            (func_ptr)SortSVEkv::CoreSmallSort14,
                            (func_ptr)SortSVEkv::CoreSmallSort15,
                            (func_ptr)SortSVEkv::CoreSmallSort16 };

        for(int idxSize = 1 ; idxSize <= 16 ; ++idxSize){
            std::cout << "Start testSort" << idxSize << "Vec_pair int...\n";
            const int Size = idxSize;
            const int VecSize = svcntb()/sizeof(int);
            srand48(0);
            const static int NbLoops = 1000;
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[Size*VecSize];

                createRandVec(vecTest, Size*VecSize);

                int values[Size*VecSize];
                for(int idxval = 0 ; idxval < Size*VecSize ; ++idxval){
                    values[idxval] = vecTest[idxval]*100+1;
                }

                Checker<int> checker(vecTest, vecTest, Size*VecSize);
                functions[Size-1](vecTest, values);
                assertNotSorted(vecTest, Size*VecSize, "testSortVec_Core_Equal");

                for(int idxval = 0 ; idxval < Size*VecSize ; ++idxval){
                    if(values[idxval] != vecTest[idxval]*100+1){
                        std::cout << "Error in testSortXVec_pair "
                                     " is " << values[idxval] <<
                                     " should be " << vecTest[idxval]*100+1 << std::endl;
                        test_res = 1;
                    }
                }
            }
        }
    }
    if(svcntb() == 512/8){

        typedef void (*func_ptr)(int*, int* );
        std::function<void(int*,int*)> functions[]
                        = { (func_ptr)SortSVEkv512::CoreSmallSort,
                            (func_ptr)SortSVEkv512::CoreSmallSort2,
                            (func_ptr)SortSVEkv512::CoreSmallSort3,
                            (func_ptr)SortSVEkv512::CoreSmallSort4,
                            (func_ptr)SortSVEkv512::CoreSmallSort5,
                            (func_ptr)SortSVEkv512::CoreSmallSort6,
                            (func_ptr)SortSVEkv512::CoreSmallSort7,
                            (func_ptr)SortSVEkv512::CoreSmallSort8,
                            (func_ptr)SortSVEkv512::CoreSmallSort9,
                            (func_ptr)SortSVEkv512::CoreSmallSort10,
                            (func_ptr)SortSVEkv512::CoreSmallSort11,
                            (func_ptr)SortSVEkv512::CoreSmallSort12,
                            (func_ptr)SortSVEkv512::CoreSmallSort13,
                            (func_ptr)SortSVEkv512::CoreSmallSort14,
                            (func_ptr)SortSVEkv512::CoreSmallSort15,
                            (func_ptr)SortSVEkv512::CoreSmallSort16 };

        for(int idxSize = 1 ; idxSize <= 16 ; ++idxSize){
            std::cout << "Start testSort" << idxSize << "Vec_pair int...\n";
            const int Size = idxSize;
            const int VecSize = svcntb()/sizeof(int);
            srand48(0);
            const static int NbLoops = 1000;
            for(int idx = 0 ; idx < NbLoops ; ++idx){
                int vecTest[Size*VecSize];

                createRandVec(vecTest, Size*VecSize);

                int values[Size*VecSize];
                for(int idxval = 0 ; idxval < Size*VecSize ; ++idxval){
                    values[idxval] = vecTest[idxval]*100+1;
                }

                Checker<int> checker(vecTest, vecTest, Size*VecSize);
                functions[Size-1](vecTest, values);
                assertNotSorted(vecTest, Size*VecSize, "testSortVec_Core_Equal");

                for(int idxval = 0 ; idxval < Size*VecSize ; ++idxval){
                    if(values[idxval] != vecTest[idxval]*100+1){
                        std::cout << "Error in testSortXVec_pair "
                                     " is " << values[idxval] <<
                                     " should be " << vecTest[idxval]*100+1 << std::endl;
                        test_res = 1;
                        assert(0);
                    }
                }
            }
        }
    }
}

template <class NumType>
void testQsSVE(){
    std::cout << "Start SortSVE sort...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        SortSVE::Sort<NumType,size_t>(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
#if defined(_OPENMP)
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        SortSVE::SortOmpPartition<NumType,size_t>(array.get(), idx);
        assertNotSorted(array.get(), idx, "");
    }
#endif
    if(svcntb() == 512/8){
        std::cout << "Start SortSVE sort512...\n";
        for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
            SortSVE512::Sort<NumType,size_t>(array.get(), idx);
            assertNotSorted(array.get(), idx, "");
        }
    #if defined(_OPENMP)
        for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
            SortSVE512::SortOmpPartition<NumType,size_t>(array.get(), idx);
            assertNotSorted(array.get(), idx, "");
        }
    #endif

    }
}

template <class NumType>
void testQsSVE_pair(){
    std::cout << "Start testQsSVE_pair...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        std::unique_ptr<NumType[]> values(new NumType[idx]);
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            values[idxval] = array[idxval]*100+1;
        }
        SortSVEkv::Sort<NumType,size_t>(array.get(), values.get(), idx);
        assertNotSorted(array.get(), idx, "");
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            if(values[idxval] != array[idxval]*100+1){
                std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                test_res = 1;
            }
        }
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        std::unique_ptr<NumType[]> values(new NumType[idx]);
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            values[idxval] = array[idxval]*100+1;
        }

        std::unique_ptr<std::pair<NumType,NumType>[]> arrayPair(new std::pair<NumType,NumType>[idx]);
        for(int idxval = 0 ; idxval < idx ; ++idxval){
            arrayPair[idxval].first = array[idxval];
            arrayPair[idxval].second = values[idxval];
        }

        SortSVEkv::Sort<std::pair<NumType,NumType>,size_t>(arrayPair.get(), idx);

        for(int idxval = 0 ; idxval < idx ; ++idxval){
            array[idxval] = arrayPair[idxval].first;
            values[idxval] = arrayPair[idxval].second;
        }

        assertNotSorted(array.get(), idx, "");
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            if(values[idxval] != array[idxval]*100+1){
                std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                test_res = 1;
            }
        }
    }
#if defined(_OPENMP)
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        std::unique_ptr<NumType[]> values(new NumType[idx]);
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            values[idxval] = array[idxval]*100+1;
        }
        SortSVEkv::SortOmpPartition<NumType,size_t>(array.get(), values.get(), idx);
        assertNotSorted(array.get(), idx, "");
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            if(values[idxval] != array[idxval]*100+1){
                std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                test_res = 1;
            }
        }
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        std::unique_ptr<NumType[]> values(new NumType[idx]);
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            values[idxval] = array[idxval]*100+1;
        }

        std::unique_ptr<std::pair<NumType,NumType>[]> arrayPair(new std::pair<NumType,NumType>[idx]);
        for(int idxval = 0 ; idxval < idx ; ++idxval){
            arrayPair[idxval].first = array[idxval];
            arrayPair[idxval].second = values[idxval];
        }

        SortSVEkv::SortOmpPartition<std::pair<NumType,NumType>,size_t>(arrayPair.get(), idx);

        for(int idxval = 0 ; idxval < idx ; ++idxval){
            array[idxval] = arrayPair[idxval].first;
            values[idxval] = arrayPair[idxval].second;
        }

        assertNotSorted(array.get(), idx, "");
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            if(values[idxval] != array[idxval]*100+1){
                std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                test_res = 1;
            }
        }
    }
#endif
    if(svcntb() == 512/8){
        std::cout << "Start testQsSVE_pair512...\n";
        for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
            std::unique_ptr<NumType[]> values(new NumType[idx]);
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                values[idxval] = array[idxval]*100+1;
            }
            SortSVEkv512::Sort<NumType,size_t>(array.get(), values.get(), idx);
            assertNotSorted(array.get(), idx, "");
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                if(values[idxval] != array[idxval]*100+1){
                    std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                    test_res = 1;
                }
            }
        }
        for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
            std::unique_ptr<NumType[]> values(new NumType[idx]);
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                values[idxval] = array[idxval]*100+1;
            }

            std::unique_ptr<std::pair<NumType,NumType>[]> arrayPair(new std::pair<NumType,NumType>[idx]);
            for(int idxval = 0 ; idxval < idx ; ++idxval){
                arrayPair[idxval].first = array[idxval];
                arrayPair[idxval].second = values[idxval];
            }

            SortSVEkv512::Sort<std::pair<NumType,NumType>,size_t>(arrayPair.get(), idx);

            for(int idxval = 0 ; idxval < idx ; ++idxval){
                array[idxval] = arrayPair[idxval].first;
                values[idxval] = arrayPair[idxval].second;
            }

            assertNotSorted(array.get(), idx, "");
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                if(values[idxval] != array[idxval]*100+1){
                    std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                    test_res = 1;
                }
            }
        }
    #if defined(_OPENMP)
        for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
            std::unique_ptr<NumType[]> values(new NumType[idx]);
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                values[idxval] = array[idxval]*100+1;
            }
            SortSVEkv512::SortOmpPartition<NumType,size_t>(array.get(), values.get(), idx);
            assertNotSorted(array.get(), idx, "");
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                if(values[idxval] != array[idxval]*100+1){
                    std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                    test_res = 1;
                }
            }
        }
        for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
            std::unique_ptr<NumType[]> values(new NumType[idx]);
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                values[idxval] = array[idxval]*100+1;
            }

            std::unique_ptr<std::pair<NumType,NumType>[]> arrayPair(new std::pair<NumType,NumType>[idx]);
            for(int idxval = 0 ; idxval < idx ; ++idxval){
                arrayPair[idxval].first = array[idxval];
                arrayPair[idxval].second = values[idxval];
            }

            SortSVEkv512::SortOmpPartition<std::pair<NumType,NumType>,size_t>(arrayPair.get(), idx);

            for(int idxval = 0 ; idxval < idx ; ++idxval){
                array[idxval] = arrayPair[idxval].first;
                values[idxval] = arrayPair[idxval].second;
            }

            assertNotSorted(array.get(), idx, "");
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                if(values[idxval] != array[idxval]*100+1){
                    std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                    test_res = 1;
                }
            }
        }
    #endif
    }
}


template <class NumType>
void testPartition(){
    std::cout << "Start SortSVE::PartitionSVE...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        const NumType pivot = NumType(idx/2);
        size_t limite = SortSVE::PartitionSVE<size_t>(&array[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
    }
    for(size_t idx = 1 ; idx <= 1000; ++idx){
        if(idx%100 == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        const NumType pivot = NumType(idx/2);
        size_t limite = SortSVE::PartitionSVE<size_t>(&array[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);

        for(size_t idxVal = 0 ; idxVal < idx ; ++idxVal){
            array[idxVal] = NumType(idx);
        }

        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        const NumType pivot = NumType(idx/2);
        size_t limite = SortSVE::PartitionSVE<size_t>(&array[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
    }
    if(svcntb() == 512/8){
        std::cout << "Start SortSVE::PartitionSVE512...\n";
        for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
            const NumType pivot = NumType(idx/2);
            size_t limite = SortSVE512::PartitionSVE<size_t>(&array[0], 0, idx-1, pivot);
            assertNotPartitioned(array.get(), idx, pivot, limite, "");
        }
        for(size_t idx = 1 ; idx <= 1000; ++idx){
            if(idx%100 == 0) std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
            const NumType pivot = NumType(idx/2);
            size_t limite = SortSVE512::PartitionSVE<size_t>(&array[0], 0, idx-1, pivot);
            assertNotPartitioned(array.get(), idx, pivot, limite, "");
        }
        for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);

            for(size_t idxVal = 0 ; idxVal < idx ; ++idxVal){
                array[idxVal] = NumType(idx);
            }

            createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
            const NumType pivot = NumType(idx/2);
            size_t limite = SortSVE512::PartitionSVE<size_t>(&array[0], 0, idx-1, pivot);
            assertNotPartitioned(array.get(), idx, pivot, limite, "");
        }

    }
}

template <class NumType>
void testPartition_pair(){
    std::cout << "Start testPartition_pair512...\n";
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        std::unique_ptr<NumType[]> values(new NumType[idx]);
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            values[idxval] = array[idxval]*100+1;
        }
        const NumType pivot = NumType(idx/2);
        size_t limite = SortSVEkv::PartitionSVE<size_t>(&array[0], &values[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            if(values[idxval] != array[idxval]*100+1){
                std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                test_res = 1;
            }
        }
    }
    for(size_t idx = 1 ; idx <= 1000; ++idx){
        if(idx%100 == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        std::unique_ptr<NumType[]> values(new NumType[idx]);
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            values[idxval] = array[idxval]*100+1;
        }
        const NumType pivot = NumType(idx/2);
        size_t limite = SortSVEkv::PartitionSVE<size_t>(&array[0], &values[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            if(values[idxval] != array[idxval]*100+1){
                std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                test_res = 1;
            }
        }
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);

        for(size_t idxVal = 0 ; idxVal < idx ; ++idxVal){
            array[idxVal] = NumType(idx);
        }

        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        std::unique_ptr<NumType[]> values(new NumType[idx]);
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            values[idxval] = array[idxval]*100+1;
        }
        const NumType pivot = NumType(idx/2);
        size_t limite = SortSVEkv::PartitionSVE<size_t>(&array[0], &values[0], 0, idx-1, pivot);
        assertNotPartitioned(array.get(), idx, pivot, limite, "");
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            if(values[idxval] != array[idxval]*100+1){
                std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                test_res = 1;
            }
        }
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        std::unique_ptr<NumType[]> values(new NumType[idx]);
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            values[idxval] = array[idxval]*100+1;
        }
        const NumType pivot = NumType(idx/2);

        std::unique_ptr<std::pair<NumType,NumType>[]> arrayPair(new std::pair<NumType,NumType>[idx]);
        for(int idxval = 0 ; idxval < idx ; ++idxval){
            arrayPair[idxval].first = array[idxval];
            arrayPair[idxval].second = values[idxval];
        }

        size_t limite = SortSVEkv::PartitionSVE<size_t>(&arrayPair[0], 0, idx-1, pivot);

        for(int idxval = 0 ; idxval < idx ; ++idxval){
            array[idxval] = arrayPair[idxval].first;
            values[idxval] = arrayPair[idxval].second;
        }

        assertNotPartitioned(array.get(), idx, pivot, limite, "");
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            if(values[idxval] != array[idxval]*100+1){
                std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                test_res = 1;
            }
        }
    }
    for(size_t idx = 1 ; idx <= 1000; ++idx){
        if(idx%100 == 0) std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);
        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        std::unique_ptr<NumType[]> values(new NumType[idx]);
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            values[idxval] = array[idxval]*100+1;
        }
        const NumType pivot = NumType(idx/2);

        std::unique_ptr<std::pair<NumType,NumType>[]> arrayPair(new std::pair<NumType,NumType>[idx]);
        for(int idxval = 0 ; idxval < idx ; ++idxval){
            arrayPair[idxval].first = array[idxval];
            arrayPair[idxval].second = values[idxval];
        }

        size_t limite = SortSVEkv::PartitionSVE<size_t>(&arrayPair[0], 0, idx-1, pivot);

        for(int idxval = 0 ; idxval < idx ; ++idxval){
            array[idxval] = arrayPair[idxval].first;
            values[idxval] = arrayPair[idxval].second;
        }

        assertNotPartitioned(array.get(), idx, pivot, limite, "");
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            if(values[idxval] != array[idxval]*100+1){
                std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                test_res = 1;
            }
        }
    }
    for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
        std::cout << "   " << idx << std::endl;
        std::unique_ptr<NumType[]> array(new NumType[idx]);

        for(size_t idxVal = 0 ; idxVal < idx ; ++idxVal){
            array[idxVal] = NumType(idx);
        }

        createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
        std::unique_ptr<NumType[]> values(new NumType[idx]);
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            values[idxval] = array[idxval]*100+1;
        }
        const NumType pivot = NumType(idx/2);

        std::unique_ptr<std::pair<NumType,NumType>[]> arrayPair(new std::pair<NumType,NumType>[idx]);
        for(int idxval = 0 ; idxval < idx ; ++idxval){
            arrayPair[idxval].first = array[idxval];
            arrayPair[idxval].second = values[idxval];
        }

        size_t limite = SortSVEkv::PartitionSVE<size_t>(&arrayPair[0], 0, idx-1, pivot);

        for(int idxval = 0 ; idxval < idx ; ++idxval){
            array[idxval] = arrayPair[idxval].first;
            values[idxval] = arrayPair[idxval].second;
        }

        assertNotPartitioned(array.get(), idx, pivot, limite, "");
        for(size_t idxval = 0 ; idxval < idx ; ++idxval){
            if(values[idxval] != array[idxval]*100+1){
                std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                test_res = 1;
            }
        }
    }
    if(svcntb() == 512/8){
        std::cout << "Start testPartition_pair512...\n";
        for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
            std::unique_ptr<NumType[]> values(new NumType[idx]);
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                values[idxval] = array[idxval]*100+1;
            }
            const NumType pivot = NumType(idx/2);
            size_t limite = SortSVEkv512::PartitionSVE<size_t>(&array[0], &values[0], 0, idx-1, pivot);
            assertNotPartitioned(array.get(), idx, pivot, limite, "");
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                if(values[idxval] != array[idxval]*100+1){
                    std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                    test_res = 1;
                }
            }
        }
        for(size_t idx = 1 ; idx <= 1000; ++idx){
            if(idx%100 == 0) std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
            std::unique_ptr<NumType[]> values(new NumType[idx]);
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                values[idxval] = array[idxval]*100+1;
            }
            const NumType pivot = NumType(idx/2);
            size_t limite = SortSVEkv512::PartitionSVE<size_t>(&array[0], &values[0], 0, idx-1, pivot);
            assertNotPartitioned(array.get(), idx, pivot, limite, "");
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                if(values[idxval] != array[idxval]*100+1){
                    std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                    test_res = 1;
                }
            }
        }
        for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);

            for(size_t idxVal = 0 ; idxVal < idx ; ++idxVal){
                array[idxVal] = NumType(idx);
            }

            createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
            std::unique_ptr<NumType[]> values(new NumType[idx]);
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                values[idxval] = array[idxval]*100+1;
            }
            const NumType pivot = NumType(idx/2);
            size_t limite = SortSVEkv512::PartitionSVE<size_t>(&array[0], &values[0], 0, idx-1, pivot);
            assertNotPartitioned(array.get(), idx, pivot, limite, "");
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                if(values[idxval] != array[idxval]*100+1){
                    std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                    test_res = 1;
                }
            }
        }
        for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
            std::unique_ptr<NumType[]> values(new NumType[idx]);
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                values[idxval] = array[idxval]*100+1;
            }
            const NumType pivot = NumType(idx/2);

            std::unique_ptr<std::pair<NumType,NumType>[]> arrayPair(new std::pair<NumType,NumType>[idx]);
            for(int idxval = 0 ; idxval < idx ; ++idxval){
                arrayPair[idxval].first = array[idxval];
                arrayPair[idxval].second = values[idxval];
            }

            size_t limite = SortSVEkv512::PartitionSVE<size_t>(&arrayPair[0], 0, idx-1, pivot);

            for(int idxval = 0 ; idxval < idx ; ++idxval){
                array[idxval] = arrayPair[idxval].first;
                values[idxval] = arrayPair[idxval].second;
            }

            assertNotPartitioned(array.get(), idx, pivot, limite, "");
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                if(values[idxval] != array[idxval]*100+1){
                    std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                    test_res = 1;
                }
            }
        }
        for(size_t idx = 1 ; idx <= 1000; ++idx){
            if(idx%100 == 0) std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
            std::unique_ptr<NumType[]> values(new NumType[idx]);
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                values[idxval] = array[idxval]*100+1;
            }
            const NumType pivot = NumType(idx/2);

            std::unique_ptr<std::pair<NumType,NumType>[]> arrayPair(new std::pair<NumType,NumType>[idx]);
            for(int idxval = 0 ; idxval < idx ; ++idxval){
                arrayPair[idxval].first = array[idxval];
                arrayPair[idxval].second = values[idxval];
            }

            size_t limite = SortSVEkv512::PartitionSVE<size_t>(&arrayPair[0], 0, idx-1, pivot);

            for(int idxval = 0 ; idxval < idx ; ++idxval){
                array[idxval] = arrayPair[idxval].first;
                values[idxval] = arrayPair[idxval].second;
            }

            assertNotPartitioned(array.get(), idx, pivot, limite, "");
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                if(values[idxval] != array[idxval]*100+1){
                    std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                    test_res = 1;
                }
            }
        }
        for(size_t idx = 1 ; idx <= (1<<10); idx *= 2){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);

            for(size_t idxVal = 0 ; idxVal < idx ; ++idxVal){
                array[idxVal] = NumType(idx);
            }

            createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);
            std::unique_ptr<NumType[]> values(new NumType[idx]);
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                values[idxval] = array[idxval]*100+1;
            }
            const NumType pivot = NumType(idx/2);

            std::unique_ptr<std::pair<NumType,NumType>[]> arrayPair(new std::pair<NumType,NumType>[idx]);
            for(int idxval = 0 ; idxval < idx ; ++idxval){
                arrayPair[idxval].first = array[idxval];
                arrayPair[idxval].second = values[idxval];
            }

            size_t limite = SortSVEkv512::PartitionSVE<size_t>(&arrayPair[0], 0, idx-1, pivot);

            for(int idxval = 0 ; idxval < idx ; ++idxval){
                array[idxval] = arrayPair[idxval].first;
                values[idxval] = arrayPair[idxval].second;
            }

            assertNotPartitioned(array.get(), idx, pivot, limite, "");
            for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                if(values[idxval] != array[idxval]*100+1){
                    std::cout << "Error in testNewPartitionSVEV2_pair, pair/key do not match" << std::endl;
                    test_res = 1;
                }
            }
        }
    }
}


template <class NumType>
void testSmallVecSort(){
    {
        std::cout << "Start SortSVE::SmallSort16V...\n";
        const size_t SizeVec = svcntb()/sizeof(NumType);
        const size_t MaxSizeAllVec = SizeVec * 16;
        for(size_t idx = 1 ; idx <= MaxSizeAllVec; idx++){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            for(int idxTest = 0 ; idxTest < 100 ; ++idxTest){
                createRandVec(array.get(), idx);
                Checker<NumType> checker(array.get(), array.get(), idx);
                SortSVE::SmallSort16V(array.get(), idx);
                assertNotSorted(array.get(), idx, "");
            }
        }
    }
    if(svcntb() == 512/8){
        std::cout << "Start SortSVE::SmallSort16V512...\n";
        const size_t SizeVec = svcntb()/sizeof(NumType);
        const size_t MaxSizeAllVec = SizeVec * 16;
        for(size_t idx = 1 ; idx <= MaxSizeAllVec; idx++){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            for(int idxTest = 0 ; idxTest < 100 ; ++idxTest){
                createRandVec(array.get(), idx);
                Checker<NumType> checker(array.get(), array.get(), idx);
                SortSVE::SmallSort16V(array.get(), idx);
                assertNotSorted(array.get(), idx, "");
            }
        }
    }
}

template <class NumType>
void testSmallVecSort_pair(){
    {
        std::cout << "Start testSmallVecSort_pair bitfull...\n";
        const size_t SizeVec = svcntb()/sizeof(NumType);
        const size_t MaxSizeAllVec = SizeVec * 16;
        for(size_t idx = 1 ; idx <= MaxSizeAllVec; idx++){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            std::unique_ptr<NumType[]> values(new NumType[idx]);
            for(size_t idxTest = 0 ; idxTest < 100 ; ++idxTest){
                createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);

                for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                    values[idxval] = array[idxval]*100+1;
                }

                SortSVEkv::SmallSort16V(array.get(), values.get(), idx);
                assertNotSorted(array.get(), idx, "");

                for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                    if(values[idxval] != array[idxval]*100+1){
                        std::cout << "Error in testSmallVecSort_pair "
                                     " is " << values[idxval] <<
                                     " should be " << array[idxval]*100+1 << std::endl;
                        test_res = 1;
                    }
                }
            }
        }
    }
    {
        const size_t SizeVec = svcntb()/sizeof(NumType);
        const size_t MaxSizeAllVec = SizeVec * 16;
        for(size_t idx = 1 ; idx <= MaxSizeAllVec; idx++){
            std::cout << "   " << idx << std::endl;
            std::unique_ptr<NumType[]> array(new NumType[idx]);
            std::unique_ptr<NumType[]> values(new NumType[idx]);
            for(size_t idxTest = 0 ; idxTest < 100 ; ++idxTest){
                createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);

                for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                    values[idxval] = array[idxval]*100+1;
                }

                std::unique_ptr<std::pair<NumType,NumType>[]> arrayPair(new std::pair<NumType,NumType>[idx]);
                for(int idxval = 0 ; idxval < idx ; ++idxval){
                    arrayPair[idxval].first = array[idxval];
                    arrayPair[idxval].second = values[idxval];
                }

                SortSVEkv::SmallSort16V(arrayPair.get(), idx);

                for(int idxval = 0 ; idxval < idx ; ++idxval){
                    array[idxval] = arrayPair[idxval].first;
                    values[idxval] = arrayPair[idxval].second;
                }

                assertNotSorted(array.get(), idx, "");

                for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                    if(values[idxval] != array[idxval]*100+1){
                        std::cout << "Error in testSmallVecSort_pair "
                                     " is " << values[idxval] <<
                                     " should be " << array[idxval]*100+1 << std::endl;
                        test_res = 1;
                    }
                }
            }
        }
    }
    if(svcntb() == 512/8){
        std::cout << "Start testSmallVecSort_pair bitfull 512...\n";
        {
            const size_t SizeVec = svcntb()/sizeof(NumType);
            const size_t MaxSizeAllVec = SizeVec * 16;
            for(size_t idx = 1 ; idx <= MaxSizeAllVec; idx++){
                std::cout << "   " << idx << std::endl;
                std::unique_ptr<NumType[]> array(new NumType[idx]);
                std::unique_ptr<NumType[]> values(new NumType[idx]);
                for(size_t idxTest = 0 ; idxTest < 100 ; ++idxTest){
                    createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);

                    for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                        values[idxval] = array[idxval]*100+1;
                    }

                    SortSVEkv512::SmallSort16V(array.get(), values.get(), idx);
                    assertNotSorted(array.get(), idx, "");

                    for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                        if(values[idxval] != array[idxval]*100+1){
                            std::cout << "Error in testSmallVecSort_pair "
                                         " is " << values[idxval] <<
                                         " should be " << array[idxval]*100+1 << std::endl;
                            test_res = 1;
                        }
                    }
                }
            }
        }
        {
            const size_t SizeVec = svcntb()/sizeof(NumType);
            const size_t MaxSizeAllVec = SizeVec * 16;
            for(size_t idx = 1 ; idx <= MaxSizeAllVec; idx++){
                std::cout << "   " << idx << std::endl;
                std::unique_ptr<NumType[]> array(new NumType[idx]);
                std::unique_ptr<NumType[]> values(new NumType[idx]);
                for(size_t idxTest = 0 ; idxTest < 100 ; ++idxTest){
                    createRandVec(array.get(), idx); Checker<NumType> checker(array.get(), array.get(), idx);

                    for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                        values[idxval] = array[idxval]*100+1;
                    }

                    std::unique_ptr<std::pair<NumType,NumType>[]> arrayPair(new std::pair<NumType,NumType>[idx]);
                    for(int idxval = 0 ; idxval < idx ; ++idxval){
                        arrayPair[idxval].first = array[idxval];
                        arrayPair[idxval].second = values[idxval];
                    }

                    SortSVEkv512::SmallSort16V(arrayPair.get(), idx);

                    for(int idxval = 0 ; idxval < idx ; ++idxval){
                        array[idxval] = arrayPair[idxval].first;
                        values[idxval] = arrayPair[idxval].second;
                    }

                    assertNotSorted(array.get(), idx, "");

                    for(size_t idxval = 0 ; idxval < idx ; ++idxval){
                        if(values[idxval] != array[idxval]*100+1){
                            std::cout << "Error in testSmallVecSort_pair "
                                         " is " << values[idxval] <<
                                         " should be " << array[idxval]*100+1 << std::endl;
                            test_res = 1;
                        }
                    }
                }
            }
        }
    }
}


template <class NumType>
void testIsSortedVec(){
    {
        std::cout << "Start testIsSortedVec...\n";
        const size_t SizeVec = svcntb()/sizeof(NumType);
        
        std::unique_ptr<NumType[]> vec(new NumType[SizeVec]);
        for(int idx = 0 ; idx < SizeVec; ++idx){
            vec[idx] = NumType(idx);
        }
        
        if(!SortSVE::IsSorted(vec.get())){
            std::cout << "Error array should be sorted " << __LINE__ << std::endl;
            test_res=1;
        }
    }
    {
        const size_t SizeVec = svcntb()/sizeof(NumType);

        std::unique_ptr<NumType[]> vec(new NumType[SizeVec]);
        for(int idx = 0 ; idx < SizeVec; ++idx){
            vec[idx] = NumType(idx%4);
        }

        if(SortSVE::IsSorted(vec.get())){
            std::cout << "Error array should NOT be sorted " << __LINE__ << std::endl;
            test_res=1;
        }
    }
    {
        const size_t SizeVec = svcntb()/sizeof(NumType);

        std::unique_ptr<NumType[]> vec(new NumType[SizeVec]);
        for(int idx = 0 ; idx < SizeVec; ++idx){
            vec[idx] = NumType((idx%6)-3);
        }

        if(SortSVE::IsSorted(vec.get())){
            std::cout << "Error array should NOT be sorted " << __LINE__ << std::endl;
            test_res=1;
        }
    }
    {
        const size_t SizeVec = svcntb()/sizeof(NumType);
        
        std::unique_ptr<NumType[]> vec(new NumType[SizeVec]);
        for(int idx = 0 ; idx < SizeVec; ++idx){
            vec[idx] = NumType(SizeVec+idx);
        }
        
        if(!SortSVE::IsSorted(vec.get())){
            std::cout << "Error array should be sorted " << __LINE__ << std::endl;
            test_res=1;
        }
    }
    {
        const size_t SizeVec = svcntb()/sizeof(NumType);
        
        std::unique_ptr<NumType[]> vec(new NumType[SizeVec]);
        for(int idx = 0 ; idx < SizeVec; ++idx){
            vec[idx] = NumType(0);
        }
        
        if(!SortSVE::IsSorted(vec.get())){
            std::cout << "Error array should be sorted " << __LINE__ << std::endl;
            test_res=1;
        }
    }
    {
        const size_t SizeVec = svcntb()/sizeof(NumType);
        
        std::unique_ptr<NumType[]> vec(new NumType[SizeVec]);
        for(int idx = 0 ; idx < SizeVec; ++idx){
            vec[idx] = NumType(SizeVec);
        }
        
        if(!SortSVE::IsSorted(vec.get())){
            std::cout << "Error array should be sorted " << __LINE__ << std::endl;
            test_res=1;
        }
    }
    {
        const size_t SizeVec = svcntb()/sizeof(NumType);
        
        std::unique_ptr<NumType[]> vec(new NumType[SizeVec]);
        for(int idx = 0 ; idx < SizeVec; ++idx){
            vec[idx] = NumType(SizeVec-idx);
        }
        
        if(SortSVE::IsSorted(vec.get())){
            std::cout << "Error array should NOT be sorted " << __LINE__ << std::endl;
            test_res=1;
        }
    }
    {
        const size_t SizeVec = svcntb()/sizeof(NumType);
        
        std::unique_ptr<NumType[]> vec(new NumType[SizeVec]);
        for(int idx = 0 ; idx < SizeVec; ++idx){
            vec[idx] = NumType(idx);
        }
        vec[0] = NumType(SizeVec*10);
        
        if(SortSVE::IsSorted(vec.get())){
            std::cout << "Error array should NOT be sorted " << __LINE__ << std::endl;
            test_res=1;
        }
    }
    {
        const size_t SizeVec = svcntb()/sizeof(NumType);
        
        std::unique_ptr<NumType[]> vec(new NumType[SizeVec]);
        for(int idx = 0 ; idx < SizeVec; ++idx){
            vec[idx] = NumType(idx);
        }
        vec[SizeVec-1] = -NumType(SizeVec);
        
        if(SortSVE::IsSorted(vec.get())){
            std::cout << "Error array should NOT be sorted " << __LINE__ << std::endl;
            test_res=1;
        }
    }
    {
        const size_t SizeVec = svcntb()/sizeof(NumType);
        
        std::unique_ptr<NumType[]> vec(new NumType[SizeVec]);
        for(int idx = 0 ; idx < SizeVec; ++idx){
            vec[idx] = NumType((idx+5)%SizeVec);
        }
        
        if(SortSVE::IsSorted(vec.get())){
            std::cout << "Error array should NOT be sorted " << __LINE__ << std::endl;
            test_res=1;
        }
    }
    {
        const size_t SizeVec = svcntb()/sizeof(NumType);

        std::unique_ptr<NumType[]> vec(new NumType[SizeVec]);
        for(int idxShift = 1 ; idxShift < SizeVec ; ++idxShift){
            for(int idx = 0 ; idx < SizeVec; ++idx){
                vec[idx] = NumType((idx+idxShift)%SizeVec);
            }

            if(SortSVE::IsSorted(vec.get())){
                std::cout << "Error array should NOT be sorted " << __LINE__ << std::endl;
                test_res=1;
            }
        }
    }
}

int main(){
    testSortVec_pair();
    testSortXVec_pair();

    testSortVec();
    testSort2Vec();
    testSort3Vec();
    testSort4Vec();
    testSort5Vec();
    testSort6Vec();
    testSort7Vec();
    testSort8Vec();

    testSort9Vec();
    testSort10Vec();
    testSort11Vec();
    testSort12Vec();
    testSort13Vec();
    testSort14Vec();
    testSort15Vec();
    testSort16Vec();

    testSmallVecSort<int>();
    testSmallVecSort<double>();

    testSmallVecSort_pair<int>();

    testQsSVE<double>();
    testQsSVE<int>();
    testQsSVE_pair<int>();

    testPartition<int>();
    testPartition<double>();
    testPartition_pair<int>();
    
    testIsSortedVec<int>();
    testIsSortedVec<double>();

    if(test_res != 0){
        std::cout << "Test failed!" << std::endl;
    }

    return test_res;
}
