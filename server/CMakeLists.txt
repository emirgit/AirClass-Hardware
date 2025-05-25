# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and language (CXX for C++)
project(AirClassServer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Boost library package
find_package(Boost 1.70 REQUIRED COMPONENTS system thread)

# Add the executable target
add_executable(server websocket_server.cpp)

# Tell CMake to look for headers in the extern/websocketpp directory
target_include_directories(server PRIVATE ${CMAKE_SOURCE_DIR}/../extern/websocketpp)

# Link the necessary libraries
target_link_libraries(server PRIVATE
    Boost::system
    Boost::thread
    pthread # POSIX threads library
)