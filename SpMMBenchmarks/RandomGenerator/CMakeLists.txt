add_executable(randomGenerator
    ${CMAKE_CURRENT_SOURCE_DIR}/randomGenerator.cpp
)

target_link_libraries(randomGenerator
    PRIVATE
        SparseMatTool
)