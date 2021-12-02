add_executable(CheckCSREqual
    ${CMAKE_CURRENT_SOURCE_DIR}/CheckCSREqual.cpp
)

target_link_libraries(CheckCSREqual
    PRIVATE
        SparseMatTool
)