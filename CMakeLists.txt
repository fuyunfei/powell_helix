cmake_minimum_required(VERSION 2.8)

project(powell_helix)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# helloworld
add_executable(powell_helix powell_helix.cc)
target_link_libraries(powell_helix ${CERES_LIBRARIES})
