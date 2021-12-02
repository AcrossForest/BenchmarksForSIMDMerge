cmake_minimum_required(VERSION 3.10)
project(SpSpInterface)


add_library(SpSpInterface INTERFACE)
target_include_directories(SpSpInterface
INTERFACE
${PROJECT_SOURCE_DIR}/include
)
target_compile_features(SpSpInterface INTERFACE cxx_std_17)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ARM")
    set(USE_ARM TRUE)
    target_compile_options(SpSpInterface INTERFACE "-march=armv8.6-a+sve")
    target_compile_definitions(SpSpInterface INTERFACE __SPSP_USE_ARM__)
    message("Build for ARM. Use real SpSp Instructions V1" )
else()
    set(USE_ARM FALSE)
    target_compile_definitions(SpSpInterface INTERFACE __SPSP_USE_X86__)
    message("Build for X86. Use fake SpSp Instructions V1 implemented using vectors.")
endif()
