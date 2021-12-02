add_library(SparseMMHeapAccum STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/SparseMMHeapAccum.cpp
)

target_link_libraries(SparseMMHeapAccum
    PUBLIC
        SparseMatTool
        Benchmarking
)

target_compile_options(SparseMMHeapAccum PRIVATE "-g")