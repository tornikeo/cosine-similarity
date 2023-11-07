cmake_minimum_required(VERSION 3.1)
project(cosine)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(OpenMP REQUIRED)


add_executable(cosine src/main.cpp)

if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()


target_link_libraries(cosine xtensor xtensor::optimize OpenMP::OpenMP_CXX)
