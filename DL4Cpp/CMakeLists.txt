project("DL4Cpp")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Armadillo REQUIRED)

add_library(DL4Cpp INTERFACE)

target_link_libraries(DL4Cpp INTERFACE Armadillo::armadillo)


target_include_directories(DL4Cpp INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(dl4cpp_test src/main.cpp)
target_link_libraries(dl4cpp_test PRIVATE DL4Cpp)



