cmake_minimum_required(VERSION 3.5)
project(main)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER "dlcc")
# 设置目标GPU架构
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --cuda-gpu-arch=dlgpuc64")
set(SDK_DIR "/dl/sdk/")
# 根据需要设置相应的编译选项
set(CXX_FLAGS -std=c++11 -fPIC -fpermissive -Wno-attributes)

# 包含的头文件目录
include_directories(${SDK_DIR}/include)
include_directories(${SDK_DIR}/include/dlnne)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(main
    ../src/operator.cpp
    ../src/util.cpp
    ../src/memorypool.cpp
    main.cpp)

link_directories(${SDK_DIR}/lib ${CMAKE_BINARY_DIR})

