cmake_minimum_required (VERSION 3.7)

project(CppConfProject)

find_package(OpenCL REQUIRED)

add_executable(CppConfProject main.cpp graph.h path_finder.h CLPathFinder.cpp CLPathFinder.h fordBellman_ver1.cl fordBellman_ver2.cl fordBellman_ver3.cl)

target_link_libraries(CppConfProject OpenCL::OpenCL)