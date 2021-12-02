add_library(SparseOldTranspose STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/SparseOldTranspose.cpp
)

target_link_libraries(SparseOldTranspose
    PUBLIC
        SparseMatTool
)