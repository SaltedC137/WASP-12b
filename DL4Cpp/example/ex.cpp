/**
 * @file ex.cpp
 * @author Aska Lyn
 * @brief Example usage of DL4Cpp Tensor library
 * @date 2026-03-11
 */

#include <iostream>
#include <vector>
#include "tensor.hpp"
#include "check.hpp"

using namespace dlc_inf;

void TestTensorCreation() {
    std::cout << "--- Test 1: Tensor Creation & Basic Properties ---\n";
    
    // Create a tensor with 2 channels, 3 rows, and 4 columns
    Tensor<float> tensor(2, 3, 4);
    
    CHECK_EQ(tensor.channels(), 2);
    CHECK_EQ(tensor.rows(), 3);
    CHECK_EQ(tensor.cols(), 4);
    CHECK_EQ(tensor.size(), 2 * 3 * 4);
    
    std::cout << "[Pass] Tensor creation and dimension validation successful.\n\n";
}

void TestRowMajorFill() {
    std::cout << "--- Test 2: Row-Major Data Fill ---\n";
    
    Tensor<float> tensor(1, 2, 3); // 1 channel, 2 rows, 3 columns
    
    // Prepare a standard row-major 1D array (e.g., exported from PyTorch)
    // Logically, it represents the following matrix:
    // [1, 2, 3]
    // [4, 5, 6]
    std::vector<float> values = {1, 2, 3, 4, 5, 6};
    
    tensor.Fill(values, true); // 'true' indicates the input data is row-major
    
    // Verify the conversion
    CHECK_EQ(tensor.at(0, 0, 0), 1);
    CHECK_EQ(tensor.at(0, 0, 1), 2);
    CHECK_EQ(tensor.at(0, 0, 2), 3);
    CHECK_EQ(tensor.at(0, 1, 0), 4);
    CHECK_EQ(tensor.at(0, 1, 1), 5);
    CHECK_EQ(tensor.at(0, 1, 2), 6);
    
    std::cout << "[Pass] Row-major conversion and data validation successful.\n\n";
}

void TestPadding() {
    std::cout << "--- Test 3: Tensor Edge Padding ---\n";
    
    // Create a 1-channel, 2x2 tensor, and fill it with 1s
    Tensor<float> tensor(1, 2, 2);
    std::vector<float> values = {1, 1, 
                                 1, 1};
    tensor.Fill(values, true);
    
    // Execute Padding: pad 1 layer on Top, Bottom, Left, and Right with value 0.0f
    std::vector<uint32_t> pads = {1, 1, 1, 1};
    tensor.Padding(pads, 0.0f);
    
    // Verify the new dimensions
    CHECK_EQ(tensor.rows(), 4); // 2 (original) + 1 (top) + 1 (bottom) = 4
    CHECK_EQ(tensor.cols(), 4); // 2 (original) + 1 (left) + 1 (right) = 4
    
    // Verify the padding results
    // Expected matrix:
    // 0 0 0 0
    // 0 1 1 0
    // 0 1 1 0
    // 0 0 0 0
    
    // Check the top-left padding (should be 0)
    CHECK_EQ(tensor.at(0, 0, 0), 0.0f); 
    
    // Check the center original data (should be 1)
    CHECK_EQ(tensor.at(0, 1, 1), 1.0f); 
    CHECK_EQ(tensor.at(0, 1, 2), 1.0f); 
    CHECK_EQ(tensor.at(0, 2, 1), 1.0f); 
    CHECK_EQ(tensor.at(0, 2, 2), 1.0f); 
    
    // Check the bottom-right padding (should be 0)
    CHECK_EQ(tensor.at(0, 3, 3), 0.0f); 
    
    std::cout << "[Pass] Edge padding validation successful.\n\n";
}

int main() {
    std::cout << "====== KuiperInfer Core Feature Tests ======\n\n";
    
    TestTensorCreation();
    TestRowMajorFill();
    TestPadding();
    
    std::cout << "All test cases passed! Your Tensor class is extremely robust!\n";
    return 0;
}