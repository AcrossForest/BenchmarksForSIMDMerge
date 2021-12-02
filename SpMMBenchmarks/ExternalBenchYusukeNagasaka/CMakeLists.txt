project(YusukeNagasaka)
add_library(YusukeNagasaka 
    ${CMAKE_CURRENT_SOURCE_DIR}/YusukeNagasaka.cpp
)

target_include_directories(YusukeNagasaka
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(YusukeNagasaka
    PUBLIC
        "-march=native"
)