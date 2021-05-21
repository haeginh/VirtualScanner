cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(tester)

find_package(PCL 1.5 REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (tester virtual_scanner.cpp)
target_link_libraries (tester ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})
