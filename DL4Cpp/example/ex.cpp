/**
 * @file ex.cpp
 * @author Aska Lyn
 * @brief Comprehensive tests for DL4Cpp Tensor library
 * @date 2026-03-11 21:37:37
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "tensor.hpp"
#include "check.hpp"

using namespace dlc_inf;

// Helper function to compare floats with tolerance
bool FloatEq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

void TestTensorCreation() {
    std::cout << "--- Test 1: Tensor Creation & Basic Properties ---\n";

    // Test 3D constructor
    Tensor<float> tensor3d(2, 3, 4);
    CHECK_EQ(tensor3d.channels(), 2);
    CHECK_EQ(tensor3d.rows(), 3);
    CHECK_EQ(tensor3d.cols(), 4);
    CHECK_EQ(tensor3d.size(), 2 * 3 * 4);
    std::cout << "[Pass] 3D tensor creation.\n";

    // Test 2D constructor
    Tensor<float> tensor2d(3, 4);
    CHECK_EQ(tensor2d.channels(), 1);
    CHECK_EQ(tensor2d.rows(), 3);
    CHECK_EQ(tensor2d.cols(), 4);
    std::cout << "[Pass] 2D tensor creation.\n";

    // Test 1D constructor
    Tensor<float> tensor1d(10);
    CHECK_EQ(tensor1d.channels(), 1);
    CHECK_EQ(tensor1d.rows(), 1);
    CHECK_EQ(tensor1d.cols(), 10);
    CHECK_EQ(tensor1d.size(), 10);
    std::cout << "[Pass] 1D tensor creation.\n";

    // Test vector constructor
    Tensor<float> tensor_vec(std::vector<uint32_t>{2, 3, 4});
    CHECK_EQ(tensor_vec.channels(), 2);
    CHECK_EQ(tensor_vec.rows(), 3);
    CHECK_EQ(tensor_vec.cols(), 4);
    std::cout << "[Pass] Vector constructor.\n";

    // Test empty()
    Tensor<float> empty_tensor;
    CHECK_EQ(empty_tensor.empty(), true);
    CHECK_EQ(tensor3d.empty(), false);
    std::cout << "[Pass] empty() check.\n\n";
}

void TestCopyAndMove() {
    std::cout << "--- Test 2: Copy and Move Semantics ---\n";

    // Test copy constructor
    Tensor<float> original(2, 2, 2);
    std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8};
    original.Fill(values, false);

    Tensor<float> copied(original);
    CHECK_EQ(copied.at(0, 0, 0), 1);
    CHECK_EQ(copied.at(1, 1, 1), 8);
    std::cout << "[Pass] Copy constructor.\n";

    // Test copy assignment
    Tensor<float> assigned(1, 1, 1);
    assigned = original;
    CHECK_EQ(assigned.at(0, 0, 0), 1);
    std::cout << "[Pass] Copy assignment.\n";

    // Test move constructor
    Tensor<float> moved(std::move(assigned));
    CHECK_EQ(moved.at(0, 0, 0), 1);
    std::cout << "[Pass] Move constructor.\n";

    // Test move assignment
    Tensor<float> move_assigned(1, 1, 1);
    move_assigned = std::move(moved);
    CHECK_EQ(move_assigned.at(0, 0, 0), 1);
    std::cout << "[Pass] Move assignment.\n\n";
}

void TestAccessors() {
    std::cout << "--- Test 3: Element Access Methods ---\n";

    // Test at() with 3D tensor
    Tensor<float> tensor3d(2, 3, 4);
    std::vector<float> values3d(24);
    for (int i = 0; i < 24; ++i) values3d[i] = static_cast<float>(i + 1);
    tensor3d.Fill(values3d, false);

    CHECK_EQ(tensor3d.at(0, 0, 0), 1);
    CHECK_EQ(tensor3d.at(1, 2, 3), 24);
    std::cout << "[Pass] at() access.\n";

    // Test posi() with 2D tensor
    Tensor<float> tensor2d(3, 4);
    std::vector<float> values2d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    tensor2d.Fill(values2d, false);  // column-major: fills down columns first

    // Column-major layout for 3x4:
    // [1,  4,  7, 10]
    // [2,  5,  8, 11]
    // [3,  6,  9, 12]
    CHECK_EQ(tensor2d.posi(0, 0), 1);
    CHECK_EQ(tensor2d.posi(1, 0), 2);
    CHECK_EQ(tensor2d.posi(2, 3), 12);
    std::cout << "[Pass] posi() access.\n";

    // Test index() - 1D linear access
    CHECK_EQ(tensor3d.index(0), 1);
    CHECK_EQ(tensor3d.index(23), 24);
    std::cout << "[Pass] index() access.\n";

    // Test mutable access
    tensor3d.at(0, 0, 0) = 999.0f;
    CHECK_EQ(tensor3d.at(0, 0, 0), 999.0f);
    std::cout << "[Pass] Mutable access.\n\n";
}

void TestDataAndSlice() {
    std::cout << "--- Test 4: data() and slice() Methods ---\n";

    Tensor<float> tensor(2, 2, 2);
    std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8};
    tensor.Fill(values, false);

    // Test data()
    const auto& data = tensor.data();
    CHECK_EQ(data.n_slices, 2);
    CHECK_EQ(data.n_rows, 2);
    CHECK_EQ(data.n_cols, 2);
    std::cout << "[Pass] data() method.\n";

    // Test slice()
    auto& slice0 = tensor.slice(0);
    CHECK_EQ(slice0(0, 0), 1);
    CHECK_EQ(slice0(1, 1), 4);
    std::cout << "[Pass] slice() method.\n\n";
}

void TestFillMethods() {
    std::cout << "--- Test 5: Fill Methods ---\n";

    // Test Fill(float value)
    Tensor<float> tensor1(2, 2, 2);
    tensor1.Fill(3.14f);
    CHECK_EQ(tensor1.at(0, 0, 0), 3.14f);
    CHECK_EQ(tensor1.at(1, 1, 1), 3.14f);
    std::cout << "[Pass] Fill(float).\n";

    // Test Fill(vector, row_major=true)
    Tensor<float> tensor2(2, 2, 2);
    std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8};
    tensor2.Fill(values, true);
    CHECK_EQ(tensor2.at(0, 0, 0), 1);
    CHECK_EQ(tensor2.at(0, 0, 1), 2);
    CHECK_EQ(tensor2.at(0, 1, 0), 3);
    CHECK_EQ(tensor2.at(0, 1, 1), 4);
    CHECK_EQ(tensor2.at(1, 0, 0), 5);
    std::cout << "[Pass] Fill(vector, row_major=true).\n";

    // Test Fill(vector, row_major=false)
    Tensor<float> tensor3(2, 2, 2);
    tensor3.Fill(values, false);
    CHECK_EQ(tensor3.at(0, 0, 0), 1);
    CHECK_EQ(tensor3.at(0, 1, 0), 2);
    CHECK_EQ(tensor3.at(0, 0, 1), 3);
    CHECK_EQ(tensor3.at(0, 1, 1), 4);
    CHECK_EQ(tensor3.at(1, 0, 0), 5);
    std::cout << "[Pass] Fill(vector, row_major=false).\n";

    // Test One()
    Tensor<float> tensor4(2, 2);
    tensor4.One();
    CHECK_EQ(tensor4.at(0, 0, 0), 1.0f);
    CHECK_EQ(tensor4.at(0, 1, 1), 1.0f);
    std::cout << "[Pass] One().\n";

    // Test Rand()
    Tensor<float> tensor5(2, 2);
    tensor5.Rand();
    CHECK_EQ(tensor5.empty(), false);
    std::cout << "[Pass] Rand().\n\n";
}

void TestValuesAndShow() {
    std::cout << "--- Test 6: values() and Show() ---\n";

    Tensor<float> tensor(2, 2, 2);
    std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8};
    tensor.Fill(values, false);  // column-major fill

    // Test values(row_major=false) - returns memory layout
    auto retrieved = tensor.values(false);
    CHECK_EQ(retrieved.size(), 8);
    for (size_t i = 0; i < 8; ++i) {
        CHECK_EQ(retrieved[i], values[i]);
    }
    std::cout << "[Pass] values(false).\n";

    // Test values(row_major=true)
    auto retrieved_row = tensor.values(true);
    CHECK_EQ(retrieved_row.size(), 8);
    // Just verify it returns correct size and first/last elements
    CHECK_EQ(retrieved_row[0], 1);
    CHECK_EQ(retrieved_row[7], 8);
    std::cout << "[Pass] values(true).\n";

    // Test Show()
    std::cout << "Show() output:\n";
    tensor.Show();
    std::cout << "[Pass] Show().\n\n";
}

void TestPadding() {
    std::cout << "--- Test 7: Padding ---\n";

    Tensor<float> tensor(1, 2, 2);
    tensor.Fill(1.0f);

    std::vector<uint32_t> pads = {1, 1, 1, 1};
    tensor.Padding(pads, 0.0f);

    CHECK_EQ(tensor.rows(), 4);
    CHECK_EQ(tensor.cols(), 4);
    CHECK_EQ(tensor.at(0, 0, 0), 0.0f);
    CHECK_EQ(tensor.at(0, 1, 1), 1.0f);
    CHECK_EQ(tensor.at(0, 3, 3), 0.0f);
    std::cout << "[Pass] Padding.\n\n";
}

void TestReshape() {
    std::cout << "--- Test 8: Reshape ---\n";

    // Test reshape to 3D
    Tensor<float> tensor1(4, 4);
    std::vector<float> values(16);
    for (int i = 0; i < 16; ++i) values[i] = static_cast<float>(i + 1);
    tensor1.Fill(values, true);

    tensor1.Reshape({2, 2, 4}, true);
    CHECK_EQ(tensor1.channels(), 2);
    CHECK_EQ(tensor1.rows(), 2);
    CHECK_EQ(tensor1.cols(), 4);
    std::cout << "[Pass] Reshape to 3D.\n";

    // Test reshape to 2D
    tensor1.Reshape({4, 4}, true);
    CHECK_EQ(tensor1.channels(), 1);
    CHECK_EQ(tensor1.rows(), 4);
    CHECK_EQ(tensor1.cols(), 4);
    std::cout << "[Pass] Reshape to 2D.\n";

    // Test reshape to 1D
    tensor1.Reshape({16}, true);
    CHECK_EQ(tensor1.size(), 16);
    std::cout << "[Pass] Reshape to 1D.\n\n";
}

void TestFlatten() {
    std::cout << "--- Test 9: Flatten ---\n";

    Tensor<float> tensor(2, 2, 2);
    std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8};
    tensor.Fill(values, false);

    tensor.Flatten(false);
    CHECK_EQ(tensor.channels(), 1);
    CHECK_EQ(tensor.rows(), 1);
    CHECK_EQ(tensor.cols(), 8);
    CHECK_EQ(tensor.index(0), 1);
    CHECK_EQ(tensor.index(7), 8);
    std::cout << "[Pass] Flatten.\n\n";
}

void TestTransform() {
    std::cout << "--- Test 10: Transform ---\n";

    Tensor<float> tensor(2, 2);
    std::vector<float> values = {-2, -1, 1, 2};
    tensor.Fill(values, false);

    // Apply ReLU
    tensor.Transform([](float x) { return std::max(0.0f, x); });

    CHECK_EQ(tensor.at(0, 0, 0), 0.0f);
    CHECK_EQ(tensor.at(0, 1, 0), 0.0f);
    CHECK_EQ(tensor.at(0, 0, 1), 1.0f);
    CHECK_EQ(tensor.at(0, 1, 1), 2.0f);
    std::cout << "[Pass] Transform (ReLU).\n\n";
}

void TestRawPointers() {
    std::cout << "--- Test 11: Raw Pointers ---\n";

    Tensor<float> tensor(2, 2, 2);
    std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8};
    tensor.Fill(values, false);

    // Test raw_ptr()
    float* ptr = tensor.raw_ptr();
    CHECK_EQ(ptr[0], 1);
    CHECK_EQ(ptr[7], 8);
    std::cout << "[Pass] raw_ptr().\n";

    // Test raw_ptr(offset)
    float* ptr_offset = tensor.raw_ptr(3);
    CHECK_EQ(ptr_offset[0], 4);
    std::cout << "[Pass] raw_ptr(offset).\n";

    // Test matrix_raw_ptr()
    float* matrix_ptr = tensor.matrix_raw_ptr(1);
    CHECK_EQ(matrix_ptr[0], 5);
    std::cout << "[Pass] matrix_raw_ptr().\n";

    // Test tensor_raw_ptr()
    float* tensor_ptr = tensor.tensor_raw_ptr(0);
    CHECK_EQ(tensor_ptr[0], 1);
    std::cout << "[Pass] tensor_raw_ptr().\n\n";
}

void TestSetData() {
    std::cout << "--- Test 12: set_data() ---\n";

    Tensor<float> tensor(2, 2, 2);
    arma::fcube new_data(2, 2, 2);
    new_data.fill(9.9f);

    tensor.set_data(new_data);
    CHECK_EQ(tensor.at(0, 0, 0), 9.9f);
    CHECK_EQ(tensor.at(1, 1, 1), 9.9f);
    std::cout << "[Pass] set_data().\n\n";
}

void TestShapes() {
    std::cout << "--- Test 13: shapes() and sub_shape() ---\n";

    Tensor<float> tensor(2, 3, 4);

    auto shapes = tensor.shapes();
    CHECK_EQ(shapes.size(), 3);
    CHECK_EQ(shapes[0], 2);
    CHECK_EQ(shapes[1], 3);
    CHECK_EQ(shapes[2], 4);
    std::cout << "[Pass] shapes().\n";

    auto sub = tensor.sub_shape();
    CHECK_EQ(sub.size(), 3);
    CHECK_EQ(sub[0], 2);
    CHECK_EQ(sub[1], 3);
    CHECK_EQ(sub[2], 4);
    std::cout << "[Pass] sub_shape().\n\n";
}

int main() {
    std::cout << "====== DL4Cpp Tensor Comprehensive Tests ======\n\n";

    TestTensorCreation();
    TestCopyAndMove();
    TestAccessors();
    TestDataAndSlice();
    TestFillMethods();
    TestValuesAndShow();
    TestPadding();
    TestReshape();
    TestFlatten();
    TestTransform();
    TestRawPointers();
    TestSetData();
    TestShapes();

    std::cout << "====== All test cases passed! ======\n";
    return 0;
}
