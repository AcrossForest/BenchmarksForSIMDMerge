project(SVESort)

add_executable(TestMain
    # ./sortSVEperf.cpp
    ./testMain.cpp
)

target_compile_options(TestMain PUBLIC "-march=armv8.2-a+sve")
target_link_libraries(TestMain
    PUBLIC "-static"
)
