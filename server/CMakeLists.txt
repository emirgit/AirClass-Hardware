cmake_minimum_required(VERSION 3.10)
project(classroom_websocket_server)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS system thread)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(server websocket_server.cpp)
target_link_libraries(server ${Boost_LIBRARIES} pthread)
