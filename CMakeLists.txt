project(RootReader)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

macro(copy_file_if_changed in_file out_file target)  
    add_custom_command (
	    TARGET     ${target}
	    POST_BUILD
	    COMMAND    ${CMAKE_COMMAND}
	    ARGS       -E copy_if_different ${in_file} ${out_file}
	    COMMENT "Copying file: ${in_file} to: ${out_file}"
    )
endmacro(copy_file_if_changed)

find_package(TensorFlow)
find_package(ROOT)

include_directories(${TensorFlow_INCLUDE_DIRS} ${ROOT_INCLUDE_DIR})
add_library(RootReader
    MODULE
    RootReader.cc
    BatchedTransformation.cc
)
target_link_libraries(RootReader ${TensorFlow_LIBRARY} ${ROOT_LIBRARIES})

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11)

copy_file_if_changed(
    ${CMAKE_SOURCE_DIR}/root_reader.py
    ${CMAKE_CURRENT_BINARY_DIR}/root_reader.py
    RootReader
)
