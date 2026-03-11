# WASP-12b Root Project
cmake_minimum_required(VERSION 3.14)

# Enable compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# vcpkg toolchain
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

project("WASP-12b" VERSION 0.1.0 LANGUAGES CXX)

# Global C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Subdirectories
add_subdirectory("DL4Cpp")
