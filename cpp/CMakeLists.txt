project(kf_code)
cmake_minimum_required(VERSION 3.10)

add_subdirectory(thirdparty/pybind11)
include_directories(thirdparty/eigen/eigen-3.4.0)

pybind11_add_module(kf_cpp wrappers.cpp kf.h)