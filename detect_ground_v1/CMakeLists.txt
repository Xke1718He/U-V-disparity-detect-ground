cmake_minimum_required(VERSION 3.15)
project(detect_ground)

set(CMAKE_CXX_STANDARD 11)
find_package( OpenCV REQUIRED )
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        include
)
SET(SOURCE_FILE
        main.cpp
        src/DetectGround.cpp
        )
add_executable(detect_ground ${SOURCE_FILE})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})