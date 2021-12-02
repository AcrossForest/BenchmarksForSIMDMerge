cmake_minimum_required(VERSION 3.16)
project(SparseMat)


# set(CMAKE_SYSTEM_PROCESSOR "ARM")
set(USE_M5_MATIC_INST yes)
set(USE_EIGEN_BENCH "Yes")
# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)


set(CMAKE_CXX_STANDARD 20)
set(JSON_BuildTests OFF CACHE INTERNAL "")
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(SpSpInterface)
add_subdirectory(spspinterfacev2)
add_subdirectory(Gem5MagicInst)
if(DEFINED USE_EIGEN_BENCH)
    add_subdirectory(eigen)
    add_subdirectory(ExternalEigen)
    add_compile_definitions(__USE_EIGEN_BENCH__)
endif()

add_subdirectory(DriverByJson)
add_subdirectory(json)

add_subdirectory(SparseMatTool)
add_subdirectory(RandomGenerator)
add_subdirectory(SparseMMHeapAccum)
add_subdirectory(SparseMMVecAccum)
add_subdirectory(SparseMMTimSortAlike)
add_subdirectory(SparseMMTimOptimize)
add_subdirectory(SparseOldTranspose)
add_subdirectory(Benchmarking)
add_subdirectory(CheckCSREqual)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ARM")
else()
    add_subdirectory(ExternalBenchYusukeNagasaka)
    set(USE_Yusuke "Yes")
endif()
add_subdirectory(Harness)

add_executable(Test test.cpp)
target_link_libraries(Test
    PRIVATE
        Benchmarking
        SparseMatTool
        # SpMMVecAccum
        SparseOldTranspose
)

add_executable(TestEigen testEigen.cpp)
target_link_libraries(TestEigen
    PRIVATE
        Benchmarking
        SparseMatTool
        eigen
)