cmake_minimum_required(VERSION 2.8.4)

project(autocar)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

link_libraries(${OpenCV_LIBS})

if(HAVE_CUDA)
  # pass include pathes to cuda_include_directories()
  set(Caffe_ALL_INCLUDE_DIRS ${Caffe_INCLUDE_DIRS})
  list(REMOVE_ITEM Caffe_ALL_INCLUDE_DIRS PRIVATE PUBLIC)
  cuda_include_directories(${Caffe_INCLUDE_DIR} ${Caffe_SRC_DIR} ${Caffe_ALL_INCLUDE_DIRS})

  # add definitions to nvcc flags directly
  set(Caffe_ALL_DEFINITIONS ${Caffe_DEFINITIONS})
  list(REMOVE_ITEM Caffe_ALL_DEFINITIONS PRIVATE PUBLIC)
  list(APPEND CUDA_NVCC_FLAGS ${Caffe_ALL_DEFINITIONS})
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)


#Can manually add the sources using the set command as follows:
#set(SOURCES TK1_DRIVER/Controller.cpp)
#set(SOURCES TK1_DRIVER/jetsonGPIO.cpp)
#set(SOURCES TK1_DRIVER/JHPWMPCA9685.cpp)
#set(SOURCES main_autocar.cpp)
#set(SOURCES Sign_regconize.cpp)

#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/*.cpp")

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_executable(autocar ${SOURCES})

target_link_libraries(autocar ${OpenCV_LIBS})

