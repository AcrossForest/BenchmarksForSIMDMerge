cmake_minimum_required(VERSION 3.10)
project(BasicProblem)
set(CMAKE_CXX_STANDARD 20)

set(USE_M5_MATIC_INST "Yes")

add_subdirectory(SpSpInterfaceV2)
add_subdirectory(SpSpInterface)
add_subdirectory(Gem5MagicInst)



add_executable(Sort
  ${CMAKE_SOURCE_DIR}/Sort/Sort.cpp
)

target_link_libraries(Sort 
  PUBLIC "-static"
    SpSpInterfaceV2
)

add_executable(SortKV
  ${CMAKE_SOURCE_DIR}/Sort/SortKV.cpp
)

target_link_libraries(SortKV
  PUBLIC "-static"
    SpSpInterfaceV2
)

add_executable(SetOperation
  ${CMAKE_SOURCE_DIR}/SetOperation/SetOperation.cpp
)

target_link_libraries(SetOperation 
  PUBLIC "-static"
    SpSpInterfaceV2
)


add_executable(JoinOp
  ${CMAKE_SOURCE_DIR}/JoinOp/JoinOp.cpp
)

target_link_libraries(JoinOp 
  PUBLIC "-static"
    SpSpInterfaceV2
)


add_executable(ComplexJoin
    ${CMAKE_CURRENT_SOURCE_DIR}/ComplexJoin/complexJoin.cpp
)

target_link_libraries(ComplexJoin 
  PUBLIC "-static"
    SpSpInterfaceV2
)