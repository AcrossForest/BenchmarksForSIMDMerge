add_library(SparseMMTimSortAlike
    ${CMAKE_CURRENT_SOURCE_DIR}/DirectSparseVectorAdd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/InstBasedSparseVectorAdd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dummy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/timmingDebug.cpp
)

target_compile_definitions(SparseMMTimSortAlike
    PUBLIC
        # __UseTimmingDebug__
        __UseHandWriteVectorCopy__
        # __MeasureSpSpVecEfficiency__
        # __OptimizeDataCopyHalf__
        __OptimizeFinalMove__
)

target_compile_options(SparseMMTimSortAlike
    PRIVATE
        "-g"
)

target_link_libraries(SparseMMTimSortAlike
    PUBLIC
        SparseMatTool
        SpSpInterface
        M5MagicInst
        Benchmarking
)