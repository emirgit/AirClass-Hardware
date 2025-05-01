cmake_minimum_required(VERSION 3.10)
project(classroom_hardware_client)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS system thread)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(hardware_client hardware_client.cpp)
target_link_libraries(hardware_client ${Boost_LIBRARIES} pthread)
