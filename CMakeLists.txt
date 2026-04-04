# WASP-12b Root Project
cmake_minimum_required(VERSION 3.14)

# Enable compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAG -O3)

# ccache support

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()



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
add_subdirectory("DL4Cpp/net")

option(DL4CPP_BUILD_DOCS "Build Doxygen documentation" OFF)

if(DL4CPP_BUILD_DOCS)
    find_package(Doxygen REQUIRED COMPONENTS dot)

    if(DOXYGEN_FOUND)

        set(DOXYGEN_OUTPUT_DIR doc)
        set(DOXYGEN_HTML_OUTPUT ${DOXYGEN_OUTPUT_DIR}/html)
        set(DOXYGEN_HHP_FILE ${DOXYGEN_HTML_OUTPUT}/index.hhp)

        configure_file(
            ${CMAKE_SOURCE_DIR}/doc/Doxyfile.in
            ${CMAKE_BINARY_DIR}/Doxyfile
            @ONLY
        )

        find_program(HHC_COMPILER hhc PATHS 
            "C:/Progra~2/HTML~1"
            "C:/PROGRA~2/HTML~1"
        )

        add_custom_target(docs
            COMMAND ${CMAKE_COMMAND} -E make_directory ${DOXYGEN_HTML_OUTPUT}
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
            COMMAND ${CMAKE_COMMAND} -E echo "Compiling CHM file..."
            COMMAND ${HHC_COMPILER} "${DOXYGEN_HTML_OUTPUT}/index.hhp"
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Generating API documentation with Doxygen and compiling CHM"
            VERBATIM
        )

        if(HHC_COMPILER)
            message(STATUS "Doxygen documentation target 'docs' is available")
            message(STATUS "HTML Help Compiler found: ${HHC_COMPILER}")
            message(STATUS "CHM file will be: ${DOXYGEN_HTML_OUTPUT}/WASP-12b_DL4Cpp.chm")
        else()
            message(STATUS "Doxygen documentation target 'docs' is available (CHM compilation disabled - hhc.exe not found)")
        endif()
    endif()
endif()