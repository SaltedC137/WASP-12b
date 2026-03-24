# DL4Cpp Library
cmake_minimum_required(VERSION 3.14)

# Find Armadillo
find_package(Armadillo REQUIRED)
# find_package(OpenCV REQUIRED)

# Collect source files from core/ and root src/
set(DL4CPP_SOURCES
    src/core/tensor.cpp
    src/core/tensor_math.cpp
    src/core/tensor_linalg.cpp
    src/nn/layer.cpp
    src/nn/layer_factory.cpp
    src/nn/param_layer.cpp
    src/utils/log.cpp
    src/pnnx/ir.cpp
    src/pnnx/store_zip.cpp
    src/runtime/rt_op.cpp
    src/runtime/rt_ir.cpp
    src/utils/layer_bench.cpp
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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/nn>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/utils>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/pnnx>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/runtime>
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
    target_compile_options(DL4Cpp PUBLIC -mavx2)
endif()


# C++ standard
if(NOT CMAKE_CXX_STANDARD)
    set_target_properties(DL4Cpp PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
    )
endif()

# Tests==============================================
option(DL4CPP_BUILD_TESTS "Build DL4Cpp tests" ON)

if(DL4CPP_BUILD_TESTS)
    enable_testing()

    # Existing tensor tests
    add_executable(test_tensor test/test_tensor.cpp)
    target_link_libraries(test_tensor PRIVATE DL4Cpp)
    add_test(NAME TensorExample COMMAND test_tensor)

    # Runtime attribute tests
    add_executable(test_attr test/test_attr.cpp)
    target_link_libraries(test_attr PRIVATE DL4Cpp)
    add_test(NAME AttrTest COMMAND test_attr)

    # Runtime parameter tests
    add_executable(test_param test/test_param.cpp)
    target_link_libraries(test_param PRIVATE DL4Cpp)
    add_test(NAME ParamTest COMMAND test_param)

    # Runtime graph/operator/operand tests
    add_executable(test_runtime test/test_runtime.cpp)
    target_link_libraries(test_runtime PRIVATE DL4Cpp)
    add_test(NAME RuntimeTest COMMAND test_runtime)

    add_executable(bench_calcu test/bench_calcu.cpp)
    target_link_libraries(bench_calcu PRIVATE DL4Cpp)
    add_test(NAME Bench COMMAND bench_calcu)

endif()


# Tests==============================================


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
