cmake_minimum_required(VERSION 3.24)
project(GPU_MVR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

# OpenMP (Conda-safe)
find_package(OpenMP REQUIRED)

add_compile_options(
    -march=native
    -fopenmp
    -ftree-vectorize
    -fexceptions
)

include_directories("include")

add_executable(build expr/build.cpp)
target_link_libraries(build OpenMP::OpenMP_CXX)

add_executable(search expr/search.cpp)
target_link_libraries(search OpenMP::OpenMP_CXX)
