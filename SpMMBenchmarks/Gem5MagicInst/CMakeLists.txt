cmake_minimum_required(VERSION 3.10)
project(M5MagicInst C ASM)


set(SelectM5OpImpl "empty")

if(DEFINED USE_M5_MATIC_INST)
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ARM")
        set(SelectM5OpImpl "aarch64")
        message("M5MagicInst: Use aarch64 magic instructions.")
    else()
        set(SelectM5OpImpl "x86")
        message("M5MagicInst: Use x86 magic instructions.")
    endif()
else()
    message("M5MagicInst: You included M5MagicInst but haven't set USE_M5_MATIC_INST")
    message("Use empty function. All M5Ops will not be effect.")
endif()

file(GLOB SOURCE_FILE src/abi/${SelectM5OpImpl}/*.*)

message("M5MagicInst: Use ${SOURCE_FILE}")


add_library(M5MagicInst STATIC
    ${SOURCE_FILE}
)
target_include_directories(M5MagicInst
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)


target_include_directories(M5MagicInst
    PRIVATE
    ${PROJECT_SOURCE_DIR}/xinclude
)
