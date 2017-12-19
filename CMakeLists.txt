project(RootReader)
cmake_minimum_required(VERSION 3.1)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
find_package(TensorFlow)
find_package(ROOT)

include_directories(${TensorFlow_INCLUDE_DIRS} ${ROOT_INCLUDE_DIR})
add_library(RootReader
    MODULE
    RootReader.cc
    
)
target_link_libraries(RootReader ${TensorFlow_LIBRARY} ${ROOT_LIBRARIES})

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
