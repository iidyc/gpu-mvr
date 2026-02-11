cmake_minimum_required(VERSION 3.24)
project(GPU_MVR LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_BUILD_TYPE Release)

add_compile_definitions(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

find_package(faiss REQUIRED)
find_package(rmm REQUIRED)
find_package(OpenMP REQUIRED)

add_compile_options(
    -fopenmp
    -march=native
    -ftree-vectorize
    -fexceptions
)

add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=68,177,186>
)

add_compile_definitions(GPU_MVR_PROFILE)
# add_compile_definitions(GPU_MVR_OVERLAP_STAGE23)

include_directories("include")

add_executable(build expr/build.cpp)
target_link_libraries(build PRIVATE faiss OpenMP::OpenMP_CXX rmm::rmm)

add_executable(search expr/search.cpp)
target_link_libraries(search PRIVATE faiss OpenMP::OpenMP_CXX rmm::rmm)

add_executable(build_cagra expr/build_cagra.cpp)
target_link_libraries(build_cagra PRIVATE faiss OpenMP::OpenMP_CXX rmm::rmm)

add_executable(gpu_search expr/gpu_search.cu)
set_target_properties(gpu_search PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_search PRIVATE faiss OpenMP::OpenMP_CXX rmm::rmm)
