project(ExternEigen)
add_library(ExternEigen 
    INTERFACE
)

target_include_directories(ExternEigen
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ExternEigen
    INTERFACE
        eigen   
)

