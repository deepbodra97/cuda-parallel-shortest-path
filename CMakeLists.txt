cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(Project LANGUAGES CXX CUDA)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# default CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)

# default build type is Debug
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

add_executable(BellmanFord BellmanFord.cu utils.cpp utils.h utils.cuh)
add_executable(Dijkstra Dijkstra.cu utils.cpp utils.h utils.cuh)
add_executable(FloydWarshall FloydWarshall.cu utils.cpp utils.cuh)


message(STATUS "build type:  ${CMAKE_BUILD_TYPE}")