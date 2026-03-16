cmake_minimum_required(VERSION 3.14)
project(deberta.cpp)

set(CMAKE_CXX_STANDARD 17)

# ggml
add_subdirectory(ggml)

# main library
add_library(deberta deberta.cpp)
target_include_directories(deberta PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(deberta ggml)

# example
add_executable(main examples/main.cpp)
target_link_libraries(main deberta ggml)