# DL4Cpp Library
cmake_minimum_required(VERSION 3.14)

# Find Armadillo
find_package(Armadillo REQUIRED)

# Collect source files from core/ and root src/
set(DL4CPP_SOURCES
    src/core/tensor.cpp
    src/core/tensor_math.cpp
    src/core/tensor_linalg.cpp
    # src/core/tensor_nn.cpp      # TODO: Implement
    src/log.cpp
)

# Create static library
add_library(DL4Cpp STATIC ${DL4CPP_SOURCES})

# Ensure output directory exists (both at configure time and build time)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/../out/bin)
add_custom_command(TARGET DL4Cpp PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/../out/bin
    COMMENT "Ensuring out/bin directory exists"
)

# Include directories
target_include_directories(DL4Cpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/core>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/core>
)

# Link Armadillo (cross-platform)
if(TARGET Armadillo::armadillo)
    target_link_libraries(DL4Cpp PUBLIC Armadillo::armadillo)
else()
    target_link_libraries(DL4Cpp PUBLIC ${ARMADILLO_LIBRARIES})
    target_include_directories(DL4Cpp PUBLIC ${ARMADILLO_INCLUDE_DIRS})
endif()



# OpenMP support

find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP! Enabling multi-threading for DL4Cpp.")
    target_link_libraries(DL4Cpp PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(DL4Cpp PUBLIC ARMA_USE_OPENMP)
endif()


# C++ standard
if(NOT CMAKE_CXX_STANDARD)
    set_target_properties(DL4Cpp PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
    )
endif()

# Tests
option(DL4CPP_BUILD_TESTS "Build DL4Cpp tests" ON)

if(DL4CPP_BUILD_TESTS)
    enable_testing()

    add_executable(dl4cpp_test example/ex.cpp)
    target_link_libraries(dl4cpp_test PRIVATE DL4Cpp)

    add_test(NAME TensorExample COMMAND dl4cpp_test)
endif()

# Benchmarks
option(DL4CPP_BUILD_BENCHMARKS "Build DL4Cpp benchmarks" ON)

if(DL4CPP_BUILD_BENCHMARKS)
    add_executable(dl4cpp_bench example/bench.cpp)
    target_link_libraries(dl4cpp_bench PRIVATE DL4Cpp)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS DL4Cpp
    EXPORT DL4CppTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install headers from include/ and include/core/
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT DL4CppTargets
    FILE DL4CppTargets.cmake
    NAMESPACE DL4Cpp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DL4Cpp
)

# Copy headers to output directory after build
add_custom_command(TARGET DL4Cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/../out/bin/include
    COMMENT "Copying headers to out/bin/include"
)
