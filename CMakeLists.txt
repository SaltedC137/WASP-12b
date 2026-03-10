cmake_minimum_required(VERSION 3.8)


if(WIN32 AND DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

project("DL4Cpp" VERSION 1.0.0 LANGUAGES CXX)

add_subdirectory("DL4Cpp")

set_target_properties(DL4Cpp PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)