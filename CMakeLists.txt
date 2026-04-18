cmake_minimum_required(VERSION 3.18)

project(sim LANGUAGES CXX CUDA)

find_package(CUDAToolkit 11.4 REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(sim main.cu)
set_target_properties(sim PROPERTIES CUDA_RUNTIME_LIBRARY None)
target_link_libraries(sim PRIVATE CUDA::cudart)
