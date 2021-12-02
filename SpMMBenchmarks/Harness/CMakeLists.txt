add_executable(Harness
    ${CMAKE_CURRENT_SOURCE_DIR}/harness.cpp
)
message("CMake Harnesse here!")
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ARM")
    message("Build static executable to run on gem5.")
    target_link_libraries(Harness PUBLIC "-static")
endif()

target_link_libraries(Harness
    PRIVATE
        Benchmarking
        SparseMatTool
        DriverByJson
        SpMMVecAccum
        SparseMMHeapAccum
        SparseMMTimSortAlike
        SparseMMTimSortOptimized
        M5MagicInst
        # YusukeNagasaka
)

if(DEFINED USE_Yusuke)
    target_link_libraries(Harness
    PRIVATE
        YusukeNagasaka
    )
    target_compile_definitions(Harness
        PRIVATE
            __USE_YUSUKE__
    )
endif()

if(DEFINED USE_EIGEN_BENCH)
    target_link_libraries(Harness
        PRIVATE
            ExternEigen
    )

endif()

