cmake_minimum_required(VERSION 3.14)

project("nncli")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address ")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(external/axon)
add_subdirectory(external/cli)

add_executable(exec
  src/mnist_generation.cpp
  src/neural_network_cli.cpp
)

target_include_directories(exec PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(exec PUBLIC axon cli)

file(COPY Assets DESTINATION "${CMAKE_BINARY_DIR}")
