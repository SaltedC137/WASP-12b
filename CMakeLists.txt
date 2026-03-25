# WASP-12b Root Project
cmake_minimum_required(VERSION 3.14)

# Enable compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAG -O3)

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



option(DL4CPP_BUILD_DOCS "Build Doxygen documentation" OFF)

if(DL4CPP_BUILD_DOCS)
    find_package(Doxygen REQUIRED COMPONENTS dot)

    if(DOXYGEN_FOUND)
        set(DOXYGEN_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/doc)

        configure_file(
            ${CMAKE_SOURCE_DIR}/doc/Doxyfile.in
            ${CMAKE_BINARY_DIR}/Doxyfile
            @ONLY
        )

        add_custom_target(docs
            COMMAND ${CMAKE_COMMAND} -E make_directory ${DOXYGEN_OUTPUT_DIR}
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )

        message(STATUS "Doxygen documentation target 'docs' is available")
        message(STATUS "Output will be in: ${DOXYGEN_OUTPUT_DIR}")
    endif()
endif()