add_library(DriverByJson STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/DriverByJson.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/safeIO.cpp
)

target_link_libraries(DriverByJson
    PRIVATE
        nlohmann_json::nlohmann_json
)

target_link_libraries(DriverByJson
    PUBLIC
        SparseMatTool
        Benchmarking
)