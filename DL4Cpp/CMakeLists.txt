# DL4Cpp Library
cmake_minimum_required(VERSION 3.14)

# Find Armadillo
find_package(Armadillo REQUIRED)

# Create static library
add_library(DL4Cpp STATIC src/tensor.cpp src/log.cpp)

# Include directories
target_include_directories(DL4Cpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link Armadillo (cross-platform)
if(TARGET Armadillo::armadillo)
    target_link_libraries(DL4Cpp PUBLIC Armadillo::armadillo)
else()
    target_link_libraries(DL4Cpp PUBLIC ${ARMADILLO_LIBRARIES})
    target_include_directories(DL4Cpp PUBLIC ${ARMADILLO_INCLUDE_DIRS})
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

# Installation
include(GNUInstallDirs)

install(TARGETS DL4Cpp
    EXPORT DL4CppTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT DL4CppTargets
    FILE DL4CppTargets.cmake
    NAMESPACE DL4Cpp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DL4Cpp
)
