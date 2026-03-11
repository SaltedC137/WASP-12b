/**
 * @file ex.cpp
 * @author Aska Lyn
 * @brief Example usage of DL4Cpp Tensor library
 * @date 2026-03-11
 */

#include <iostream>
#include <cassert>
#include "tensor.hpp"

using namespace dlc_inf;

int main() {
    std::cout << "=== DL4Cpp Tensor Library Test ===" << std::endl;

    // 1. 测试 1D 张量
    std::cout << "\n--- Test 1D Tensor ---" << std::endl;
    Tensor<float> vec(10);
    std::cout << "Created 1D tensor, size: " << vec.size() << std::endl;
    assert(vec.size() == 10);


    // 2. 测试 2D 张量构造
    std::cout << "\n--- Test 2D Tensor ---" << std::endl;
    Tensor<float> mat(3, 4);  // 3 行 4 列
    std::cout << "Created 2D tensor: " << mat.rows() << "x" << mat.cols() << std::endl;
    assert(mat.rows() == 3);
    assert(mat.cols() == 4);


    // 3. 测试 3D 张量
    std::cout << "\n--- Test 3D Tensor ---" << std::endl;
    Tensor<float> tensor(3, 224, 224);  // RGB 图像
    std::cout << "Created 3D tensor: " << tensor.channels() 
              << "x" << tensor.rows() << "x" << tensor.cols() << std::endl;
    assert(tensor.channels() == 3);

    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
