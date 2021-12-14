# add_executable(SparseMMTimSortOptimized
#     ${CMAKE_CURRENT_SOURCE_DIR}/dummy.cpp
# )
# set(__Measure_Merge__ "Yes")

add_library(SparseMMTimSortOptimized
    INTERFACE
    # ${CMAKE_CURRENT_SOURCE_DIR}/dummy.cpp
)

target_link_libraries(SparseMMTimSortOptimized
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(SparseMMTimSortOptimized
    INTERFACE
        SpSpInterface
        SpSpInterfaceV2
        SparseMatTool
        Benchmarking
        M5MagicInst
)

if(DEFINED __Measure_Merge__)
    message("Measure the percentage of merge!")
    target_compile_definitions(SparseMMTimSortOptimized
        INTERFACE
            __Measure_Merge__
    )

    target_link_libraries(SparseMMTimSortOptimized
        INTERFACE
            M5MagicInst
    )
endif()