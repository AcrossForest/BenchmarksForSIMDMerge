add_library(SpMMVecAccum STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/SpMMVecAccum.cpp
)
target_link_libraries(SpMMVecAccum 
    PUBLIC
        SparseMatTool
        SparseOldTranspose
        Benchmarking
)
