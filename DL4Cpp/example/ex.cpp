/**
 * @file ex.cpp
 * @author Aska Lyn
 * @brief Comprehensive tests for DL4Cpp Tensor library
 * @date 2026-03-11 21:37:37
 */

#include "utils/check.hpp"
#include "core/tensor.hpp"
#include "core/tensor_linalg.hpp"
#include "core/tensor_math.hpp"
#include <cmath>
#include <iostream>
#include <vector>

using namespace ctl;
using namespace ctl::math;
using namespace ctl::linalg;

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
  for (int i = 0; i < 24; ++i)
    values3d[i] = static_cast<float>(i + 1);
  tensor3d.Fill(values3d, false);

  CHECK_EQ(tensor3d.at(0, 0, 0), 1);
  CHECK_EQ(tensor3d.at(1, 2, 3), 24);
  std::cout << "[Pass] at() access.\n";

  // Test posi() with 2D tensor
  Tensor<float> tensor2d(3, 4);
  std::vector<float> values2d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  tensor2d.Fill(values2d, false); // column-major: fills down columns first

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
  const auto &data = tensor.data();
  CHECK_EQ(data.n_slices, 2);
  CHECK_EQ(data.n_rows, 2);
  CHECK_EQ(data.n_cols, 2);
  std::cout << "[Pass] data() method.\n";

  // Test slice()
  auto &slice0 = tensor.slice(0);
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
  tensor.Fill(values, false); // column-major fill

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
  for (int i = 0; i < 16; ++i)
    values[i] = static_cast<float>(i + 1);
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
  float *ptr = tensor.raw_ptr();
  CHECK_EQ(ptr[0], 1);
  CHECK_EQ(ptr[7], 8);
  std::cout << "[Pass] raw_ptr().\n";

  // Test raw_ptr(offset)
  float *ptr_offset = tensor.raw_ptr(3);
  CHECK_EQ(ptr_offset[0], 4);
  std::cout << "[Pass] raw_ptr(offset).\n";

  // Test matrix_raw_ptr()
  float *matrix_ptr = tensor.matrix_raw_ptr(1);
  CHECK_EQ(matrix_ptr[0], 5);
  std::cout << "[Pass] matrix_raw_ptr().\n";

  // Test tensor_raw_ptr()
  float *tensor_ptr = tensor.tensor_raw_ptr(0);
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

void TestTensorAdd() {
  std::cout << "--- Test 14: Tensor Addition (add) ---\n";

  // Test 1: Element-wise addition with functional interface
  Tensor<float> tensor1(2, 2, 2);
  std::vector<float> values1 = {1, 2, 3, 4, 5, 6, 7, 8};
  tensor1.Fill(values1, false);

  Tensor<float> tensor2(2, 2, 2);
  std::vector<float> values2 = {8, 7, 6, 5, 4, 3, 2, 1};
  tensor2.Fill(values2, false);

  // Test functional add(tensor1, tensor2)
  auto result = add(tensor1, tensor2);
  CHECK_EQ(result->at(0, 0, 0), 9.0f);
  CHECK_EQ(result->at(0, 1, 1), 9.0f);
  CHECK_EQ(result->at(1, 0, 0), 9.0f);
  CHECK_EQ(result->at(1, 1, 1), 9.0f);
  std::cout << "[Pass] add(tensor1, tensor2) - element-wise.\n";

  // Test 2: Broadcasting addition (3D tensor + per-channel bias)
  Tensor<float> tensor3d(3, 2, 2);
  std::vector<float> values3d(12);
  for (int i = 0; i < 12; ++i)
    values3d[i] = static_cast<float>(i + 1);
  tensor3d.Fill(values3d, false);

  // Bias: one scalar per channel (3 channels, each 1x1)
  Tensor<float> bias(3, 1, 1); // 3 channels, 1 row, 1 col
  std::vector<float> bias_values = {100, 200, 300};
  bias.Fill(bias_values, false);

  auto result_broadcast = add(tensor3d, bias);
  // Channel 0: values [1,2,3,4] + bias[0]=100
  CHECK_EQ(result_broadcast->at(0, 0, 0), 101.0f);
  CHECK_EQ(result_broadcast->at(0, 1, 1), 104.0f);
  // Channel 1: values [5,6,7,8] + bias[1]=200
  CHECK_EQ(result_broadcast->at(1, 0, 0), 205.0f);
  CHECK_EQ(result_broadcast->at(1, 1, 1), 208.0f);
  // Channel 2: values [9,10,11,12] + bias[2]=300
  CHECK_EQ(result_broadcast->at(2, 0, 0), 309.0f);
  CHECK_EQ(result_broadcast->at(2, 1, 1), 312.0f);
  std::cout << "[Pass] add(tensor3d, bias) - broadcasting.\n\n";
}

void TestTensorSubtract() {
  std::cout << "--- Test 15: Tensor Subtraction (sub) ---\n";

  // Test 1: Element-wise subtraction with functional interface
  Tensor<float> tensor1(2, 2, 2);
  std::vector<float> values1 = {10, 20, 30, 40, 50, 60, 70, 80};
  tensor1.Fill(values1, false);

  Tensor<float> tensor2(2, 2, 2);
  std::vector<float> values2 = {1, 2, 3, 4, 5, 6, 7, 8};
  tensor2.Fill(values2, false);

  // Test functional sub(tensor1, tensor2)
  auto result = sub(tensor1, tensor2);
  CHECK_EQ(result->at(0, 0, 0), 9.0f);   // 10 - 1
  CHECK_EQ(result->at(0, 1, 1), 36.0f);  // 40 - 4
  CHECK_EQ(result->at(1, 0, 0), 45.0f);  // 50 - 5
  CHECK_EQ(result->at(1, 1, 1), 72.0f);  // 80 - 8
  std::cout << "[Pass] sub(tensor1, tensor2) - element-wise.\n";

  // Test 2: Broadcasting subtraction (3D tensor - per-channel bias)
  Tensor<float> tensor3d(3, 2, 2);
  std::vector<float> values3d(12);
  for (int i = 0; i < 12; ++i)
    values3d[i] = static_cast<float>(i + 1);
  tensor3d.Fill(values3d, false);

  // Bias: one scalar per channel (3 channels, each 1x1)
  Tensor<float> bias(3, 1, 1);
  std::vector<float> bias_values = {10, 20, 30};
  bias.Fill(bias_values, false);

  auto result_broadcast = sub(tensor3d, bias);
  // Channel 0: values [1,2,3,4] - bias[0]=10
  CHECK_EQ(result_broadcast->at(0, 0, 0), -9.0f);
  CHECK_EQ(result_broadcast->at(0, 1, 1), -6.0f);
  // Channel 1: values [5,6,7,8] - bias[1]=20
  CHECK_EQ(result_broadcast->at(1, 0, 0), -15.0f);
  CHECK_EQ(result_broadcast->at(1, 1, 1), -12.0f);
  // Channel 2: values [9,10,11,12] - bias[2]=30
  CHECK_EQ(result_broadcast->at(2, 0, 0), -21.0f);
  CHECK_EQ(result_broadcast->at(2, 1, 1), -18.0f);
  std::cout << "[Pass] sub(tensor3d, bias) - broadcasting.\n";

  // Test 3: Subtraction with negative values
  Tensor<float> neg_tensor(2, 2);
  std::vector<float> neg_values = {-5, -3, 3, 5};
  neg_tensor.Fill(neg_values, false);

  Tensor<float> pos_tensor(2, 2);
  std::vector<float> pos_values = {1, 2, 3, 4};
  pos_tensor.Fill(pos_values, false);

  auto neg_result = sub(neg_tensor, pos_tensor);
  CHECK_EQ(neg_result->at(0, 0, 0), -6.0f);   // -5 - 1
  CHECK_EQ(neg_result->at(0, 1, 0), -5.0f);   // -3 - 2
  CHECK_EQ(neg_result->at(0, 0, 1), 0.0f);    // 3 - 3
  CHECK_EQ(neg_result->at(0, 1, 1), 1.0f);    // 5 - 4
  std::cout << "[Pass] sub() with negative values.\n\n";
}

void TestTensorMultiply() {
  std::cout << "--- Test 16: Tensor Element-wise Multiplication (mul) ---\n";

  // Test 1: Element-wise multiplication with functional interface
  Tensor<float> tensor1(2, 2, 2);
  std::vector<float> values1 = {1, 2, 3, 4, 5, 6, 7, 8};
  tensor1.Fill(values1, false);

  Tensor<float> tensor2(2, 2, 2);
  std::vector<float> values2 = {8, 7, 6, 5, 4, 3, 2, 1};
  tensor2.Fill(values2, false);

  // Test functional mul(tensor1, tensor2)
  auto result = mul(tensor1, tensor2);
  CHECK_EQ(result->at(0, 0, 0), 8.0f);    // 1 * 8
  CHECK_EQ(result->at(0, 1, 1), 20.0f);   // 4 * 5
  CHECK_EQ(result->at(1, 0, 0), 20.0f);   // 5 * 4
  CHECK_EQ(result->at(1, 1, 1), 8.0f);    // 8 * 1
  std::cout << "[Pass] mul(tensor1, tensor2) - element-wise.\n";

  // Test 2: Broadcasting multiplication (3D tensor * per-channel scale)
  Tensor<float> tensor3d(3, 2, 2);
  std::vector<float> values3d(12);
  for (int i = 0; i < 12; ++i)
    values3d[i] = static_cast<float>(i + 1);
  tensor3d.Fill(values3d, false);

  // Scale: one scalar per channel (3 channels, each 1x1)
  Tensor<float> scale(3, 1, 1);
  std::vector<float> scale_values = {10, 20, 30};
  scale.Fill(scale_values, false);

  auto result_broadcast = mul(tensor3d, scale);
  // Channel 0: values [1,2,3,4] * scale[0]=10
  CHECK_EQ(result_broadcast->at(0, 0, 0), 10.0f);
  CHECK_EQ(result_broadcast->at(0, 1, 1), 40.0f);
  // Channel 1: values [5,6,7,8] * scale[1]=20
  CHECK_EQ(result_broadcast->at(1, 0, 0), 100.0f);
  CHECK_EQ(result_broadcast->at(1, 1, 1), 160.0f);
  // Channel 2: values [9,10,11,12] * scale[2]=30
  CHECK_EQ(result_broadcast->at(2, 0, 0), 270.0f);
  CHECK_EQ(result_broadcast->at(2, 1, 1), 360.0f);
  std::cout << "[Pass] mul(tensor3d, scale) - broadcasting.\n";

  // Test 3: Multiplication with zero
  Tensor<float> zero_tensor(2, 2, 2);
  zero_tensor.Fill(0.0f);
  auto zero_result = mul(tensor1, zero_tensor);
  CHECK_EQ(zero_result->at(0, 0, 0), 0.0f);
  CHECK_EQ(zero_result->at(1, 1, 1), 0.0f);
  std::cout << "[Pass] mul() with zero.\n\n";
}

void TestTensorDivide() {
  std::cout << "--- Test 17: Tensor Element-wise Division (div) ---\n";

  // Test 1: Element-wise division with functional interface
  Tensor<float> tensor1(2, 2, 2);
  std::vector<float> values1 = {8, 16, 24, 32, 40, 48, 56, 64};
  tensor1.Fill(values1, false);

  Tensor<float> tensor2(2, 2, 2);
  std::vector<float> values2 = {1, 2, 3, 4, 5, 6, 7, 8};
  tensor2.Fill(values2, false);

  // Test functional div(tensor1, tensor2)
  auto result = div(tensor1, tensor2);
  CHECK_EQ(result->at(0, 0, 0), 8.0f);    // 8 / 1
  CHECK_EQ(result->at(0, 1, 1), 8.0f);    // 32 / 4
  CHECK_EQ(result->at(1, 0, 0), 8.0f);    // 40 / 5
  CHECK_EQ(result->at(1, 1, 1), 8.0f);    // 64 / 8
  std::cout << "[Pass] div(tensor1, tensor2) - element-wise.\n";

  // Test 2: Broadcasting division (3D tensor / per-channel divisor)
  Tensor<float> tensor3d(3, 2, 2);
  std::vector<float> values3d(12);
  for (int i = 0; i < 12; ++i)
    values3d[i] = static_cast<float>(i + 1);
  tensor3d.Fill(values3d, false);

  // Divisor: one scalar per channel (3 channels, each 1x1)
  Tensor<float> divisor(3, 1, 1);
  std::vector<float> divisor_values = {1.0f, 2.0f, 4.0f};
  divisor.Fill(divisor_values, false);

  auto result_broadcast = div(tensor3d, divisor);
  // Channel 0: values [1,2,3,4] / divisor[0]=1
  CHECK_EQ(result_broadcast->at(0, 0, 0), 1.0f);
  CHECK_EQ(result_broadcast->at(0, 1, 1), 4.0f);
  // Channel 1: values [5,6,7,8] / divisor[1]=2
  CHECK_EQ(result_broadcast->at(1, 0, 0), 2.5f);
  CHECK_EQ(result_broadcast->at(1, 1, 1), 4.0f);
  // Channel 2: values [9,10,11,12] / divisor[2]=4
  CHECK_EQ(result_broadcast->at(2, 0, 0), 2.25f);
  CHECK_EQ(result_broadcast->at(2, 1, 1), 3.0f);
  std::cout << "[Pass] div(tensor3d, divisor) - broadcasting.\n";

  // Test 3: Division resulting in fractions
  Tensor<float> ones(2, 2, 2);
  ones.Fill(1.0f);
  Tensor<float> twos(2, 2, 2);
  twos.Fill(2.0f);
  auto fraction_result = div(ones, twos);
  CHECK_EQ(fraction_result->at(0, 0, 0), 0.5f);
  CHECK_EQ(fraction_result->at(1, 1, 1), 0.5f);
  std::cout << "[Pass] div() resulting in fractions.\n\n";
}

void TestScalarAdd() {
  std::cout << "--- Test 18: Scalar Addition (add with scalar) ---\n";

  Tensor<float> tensor(2, 2, 2);
  std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8};
  tensor.Fill(values, false);

  auto result = add(tensor, 10.0f);
  CHECK_EQ(result->at(0, 0, 0), 11.0f);
  CHECK_EQ(result->at(0, 1, 1), 14.0f);
  CHECK_EQ(result->at(1, 1, 1), 18.0f);
  std::cout << "[Pass] add(tensor, scalar).\n\n";
}

void TestScalarSub() {
  std::cout << "--- Test 19: Scalar Subtraction (sub with scalar) ---\n";

  Tensor<float> tensor(2, 2, 2);
  std::vector<float> values = {10, 20, 30, 40, 50, 60, 70, 80};
  tensor.Fill(values, false);

  auto result = sub(tensor, 5.0f);
  CHECK_EQ(result->at(0, 0, 0), 5.0f);
  CHECK_EQ(result->at(0, 1, 1), 35.0f);
  CHECK_EQ(result->at(1, 1, 1), 75.0f);
  std::cout << "[Pass] sub(tensor, scalar).\n\n";
}

void TestScalarMul() {
  std::cout << "--- Test 20: Scalar Multiplication (mul with scalar) ---\n";

  Tensor<float> tensor(2, 2, 2);
  std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8};
  tensor.Fill(values, false);

  auto result = mul(tensor, 3.0f);
  CHECK_EQ(result->at(0, 0, 0), 3.0f);
  CHECK_EQ(result->at(0, 1, 1), 12.0f);
  CHECK_EQ(result->at(1, 1, 1), 24.0f);
  std::cout << "[Pass] mul(tensor, scalar).\n\n";
}

void TestScalarDiv() {
  std::cout << "--- Test 21: Scalar Division (div with scalar) ---\n";

  Tensor<float> tensor(2, 2, 2);
  std::vector<float> values = {10, 20, 30, 40, 50, 60, 70, 80};
  tensor.Fill(values, false);

  auto result = div(tensor, 2.0f);
  CHECK_EQ(result->at(0, 0, 0), 5.0f);
  CHECK_EQ(result->at(0, 1, 1), 20.0f);
  CHECK_EQ(result->at(1, 1, 1), 40.0f);
  std::cout << "[Pass] div(tensor, scalar).\n\n";
}

void TestElementExp() {
  std::cout << "--- Test 22: Element-wise Exponential (exp) ---\n";

  Tensor<float> tensor(2, 2);
  std::vector<float> values = {0.0f, 1.0f, 2.0f, 3.0f};
  tensor.Fill(values, false);  // column-major: |0 2| |1 3|

  auto result = exp(tensor);
  CHECK(FloatEq(result->at(0, 0, 0), std::exp(0.0f)));  // e^0 = 1
  CHECK(FloatEq(result->at(0, 1, 0), std::exp(1.0f)));  // e^1 ≈ 2.718
  CHECK(FloatEq(result->at(0, 0, 1), std::exp(2.0f)));  // e^2 ≈ 7.389
  CHECK(FloatEq(result->at(0, 1, 1), std::exp(3.0f)));  // e^3 ≈ 20.086
  std::cout << "[Pass] exp(tensor).\n\n";
}

void TestElementClip() {
  std::cout << "--- Test 23: Element-wise Clip (clip) ---\n";

  Tensor<float> tensor(2, 2);
  std::vector<float> values = {-5.0f, 0.5f, 1.5f, 10.0f};
  tensor.Fill(values, false);  // column-major: |-5  1.5| |0.5  10|

  auto result = clip(tensor, 0.0f, 1.0f);
  CHECK_EQ(result->at(0, 0, 0), 0.0f);   // -5 -> 0
  CHECK_EQ(result->at(0, 1, 0), 0.5f);   // 0.5 -> 0.5
  CHECK_EQ(result->at(0, 0, 1), 1.0f);   // 1.5 -> 1
  CHECK_EQ(result->at(0, 1, 1), 1.0f);   // 10 -> 1
  std::cout << "[Pass] clip(tensor, min, max).\n\n";
}

void TestBatchDeterminant() {
  std::cout << "--- Test 24: Batch Determinant (det) ---\n";

  // Test single 2x2 matrix
  Tensor<float> mat1(1, 2, 2);
  // |1 3|
  // |2 4|  det = 1*4 - 2*3 = -2
  std::vector<float> values1 = {1, 2, 3, 4};
  mat1.Fill(values1, false);

  auto det_result1 = det(mat1);
  CHECK_EQ(det_result1->channels(), 1);
  CHECK(FloatEq(det_result1->at(0, 0, 0), -2.0f));
  std::cout << "[Pass] det() single matrix.\n";

  // Test batch of 3 2x2 matrices
  Tensor<float> mat_batch(3, 2, 2);
  // Channel 0: |1 3| |2 4| det = -2
  // Channel 1: |2 0| |0 2| det = 4
  // Channel 2: |1 2| |3 4| det = -2
  std::vector<float> batch_values = {1, 2, 3, 4,  2, 0, 0, 2,  1, 3, 2, 4};
  mat_batch.Fill(batch_values, false);

  auto det_result_batch = det(mat_batch);
  CHECK_EQ(det_result_batch->channels(), 3);
  CHECK(FloatEq(det_result_batch->at(0, 0, 0), -2.0f));
  CHECK(FloatEq(det_result_batch->at(1, 0, 0), 4.0f));
  CHECK(FloatEq(det_result_batch->at(2, 0, 0), -2.0f));
  std::cout << "[Pass] det() batch matrices.\n\n";
}

void TestBatchTrace() {
  std::cout << "--- Test 25: Batch Trace (trace) ---\n";

  // Test single 3x3 matrix
  Tensor<float> mat1(1, 3, 3);
  // |1 2 3|
  // |4 5 6|
  // |7 8 9|  trace = 1 + 5 + 9 = 15
  std::vector<float> values1 = {1, 4, 7, 2, 5, 8, 3, 6, 9};
  mat1.Fill(values1, false);

  auto trace_result1 = trace(mat1);
  CHECK_EQ(trace_result1->channels(), 1);
  CHECK(FloatEq(trace_result1->at(0, 0, 0), 15.0f));
  std::cout << "[Pass] trace() single matrix.\n";

  // Test batch of 2 3x3 matrices
  Tensor<float> mat_batch(2, 3, 3);
  // Channel 0: diagonal = 1, 5, 9 -> trace = 15
  // Channel 1: diagonal = 2, 4, 6 -> trace = 12
  // Column-major fill for 3x3:
  // Channel 0: |1 2 3| |4 5 6| |7 8 9| -> diagonal: 1, 5, 9
  // Channel 1: |2 3 4| |5 4 6| |7 8 6| -> diagonal: 2, 4, 6
  std::vector<float> batch_values = {1, 4, 7, 2, 5, 8, 3, 6, 9,
                                     2, 5, 7, 3, 4, 8, 4, 6, 6};
  mat_batch.Fill(batch_values, false);

  auto trace_result_batch = trace(mat_batch);
  CHECK_EQ(trace_result_batch->channels(), 2);
  CHECK(FloatEq(trace_result_batch->at(0, 0, 0), 15.0f));
  CHECK(FloatEq(trace_result_batch->at(1, 0, 0), 12.0f));
  std::cout << "[Pass] trace() batch matrices.\n\n";
}

void TestBatchInverse() {
  std::cout << "--- Test 26: Batch Inverse (inv) ---\n";

  // Test single 2x2 matrix
  Tensor<float> mat1(1, 2, 2);
  // |4 7|
  // |2 6|  det = 24-14 = 10, inv = (1/10) * |6 -7| |-2 4|
  std::vector<float> values1 = {4, 2, 7, 6};
  mat1.Fill(values1, false);

  auto inv_result1 = inv(mat1);
  CHECK_EQ(inv_result1->channels(), mat1.channels());
  CHECK_EQ(inv_result1->rows(), mat1.rows());
  CHECK_EQ(inv_result1->cols(), mat1.cols());
  // Inverse matrix: | 0.6 -0.7|
  //                 |-0.2  0.4|
  // Column-major: {0.6, -0.2, -0.7, 0.4}
  CHECK(FloatEq(inv_result1->at(0, 0, 0), 0.6f));   // (row=0, col=0)
  CHECK(FloatEq(inv_result1->at(0, 1, 0), -0.2f));  // (row=1, col=0)
  CHECK(FloatEq(inv_result1->at(0, 0, 1), -0.7f));  // (row=0, col=1)
  CHECK(FloatEq(inv_result1->at(0, 1, 1), 0.4f));   // (row=1, col=1)
  std::cout << "[Pass] inv() single matrix.\n";

  // Test batch of 2 2x2 matrices
  Tensor<float> mat_batch(2, 2, 2);
  // Channel 0: |1 0| |0 1| (identity, inv = itself)
  // Channel 1: |2 0| |0 2| inv = |0.5 0| |0 0.5|
  std::vector<float> batch_values = {1, 0, 0, 1, 2, 0, 0, 2};
  mat_batch.Fill(batch_values, false);

  auto inv_result_batch = inv(mat_batch);
  CHECK_EQ(inv_result_batch->channels(), 2);
  // Channel 0: identity
  CHECK(FloatEq(inv_result_batch->at(0, 0, 0), 1.0f));
  CHECK(FloatEq(inv_result_batch->at(0, 1, 0), 0.0f));
  CHECK(FloatEq(inv_result_batch->at(0, 0, 1), 0.0f));
  CHECK(FloatEq(inv_result_batch->at(0, 1, 1), 1.0f));
  // Channel 1: 0.5 * identity
  CHECK(FloatEq(inv_result_batch->at(1, 0, 0), 0.5f));
  CHECK(FloatEq(inv_result_batch->at(1, 1, 0), 0.0f));
  CHECK(FloatEq(inv_result_batch->at(1, 0, 1), 0.0f));
  CHECK(FloatEq(inv_result_batch->at(1, 1, 1), 0.5f));
  std::cout << "[Pass] inv() batch matrices.\n\n";
}

void TestBatchTranspose() {
  std::cout << "--- Test 27: Batch Transpose (transpose) ---\n";

  // Test single 2x3 matrix
  Tensor<float> mat1(1, 2, 3);
  // |1 2 3|
  // |4 5 6|
  // Column-major: {1, 4, 2, 5, 3, 6}
  std::vector<float> values1 = {1, 4, 2, 5, 3, 6};
  mat1.Fill(values1, false);

  auto trans_result1 = transpose(mat1);
  CHECK_EQ(trans_result1->channels(), 1);
  CHECK_EQ(trans_result1->rows(), 3);
  CHECK_EQ(trans_result1->cols(), 2);
  // Transposed: |1 4|
  //             |2 5|
  //             |3 6|
  // Column-major: {1, 2, 3, 4, 5, 6}
  CHECK_EQ(trans_result1->at(0, 0, 0), 1.0f);
  CHECK_EQ(trans_result1->at(0, 1, 0), 2.0f);
  CHECK_EQ(trans_result1->at(0, 2, 0), 3.0f);
  CHECK_EQ(trans_result1->at(0, 0, 1), 4.0f);
  CHECK_EQ(trans_result1->at(0, 1, 1), 5.0f);
  CHECK_EQ(trans_result1->at(0, 2, 1), 6.0f);
  std::cout << "[Pass] transpose() single matrix.\n";

  // Test batch of 2 2x2 matrices
  Tensor<float> mat_batch(2, 2, 2);
  // Channel 0: |1 2| |3 4| -> transpose -> |1 3| |2 4|
  // Channel 1: |5 6| |7 8| -> transpose -> |5 7| |6 8|
  // Column-major fill:
  // Channel 0: {1, 3, 2, 4} -> matrix |1 2| |3 4| -> transpose -> |1 3| |2 4| -> {1, 2, 3, 4}
  // Channel 1: {5, 7, 6, 8} -> matrix |5 6| |7 8| -> transpose -> |5 7| |6 8| -> {5, 6, 7, 8}
  std::vector<float> batch_values = {1, 3, 2, 4, 5, 7, 6, 8};
  mat_batch.Fill(batch_values, false);

  auto trans_result_batch = transpose(mat_batch);
  CHECK_EQ(trans_result_batch->channels(), 2);
  CHECK_EQ(trans_result_batch->rows(), 2);
  CHECK_EQ(trans_result_batch->cols(), 2);
  // Channel 0: transposed |1 3| |2 4|, column-major: {1, 2, 3, 4}
  CHECK_EQ(trans_result_batch->at(0, 0, 0), 1.0f);
  CHECK_EQ(trans_result_batch->at(0, 1, 0), 2.0f);
  CHECK_EQ(trans_result_batch->at(0, 0, 1), 3.0f);
  CHECK_EQ(trans_result_batch->at(0, 1, 1), 4.0f);
  // Channel 1: transposed |5 7| |6 8|, column-major: {5, 6, 7, 8}
  CHECK_EQ(trans_result_batch->at(1, 0, 0), 5.0f);
  CHECK_EQ(trans_result_batch->at(1, 1, 0), 6.0f);
  CHECK_EQ(trans_result_batch->at(1, 0, 1), 7.0f);
  CHECK_EQ(trans_result_batch->at(1, 1, 1), 8.0f);
  std::cout << "[Pass] transpose() batch matrices.\n\n";
}

void TestNormsAndDot() {
  std::cout << "--- Test 28: Norms and Dot Product ---\n";

  // Test Euclidean norm
  Tensor<float> tensor1(2, 2);
  std::vector<float> values1 = {3, 4, 0, 0};
  tensor1.Fill(values1, false);
  float norm_val = norm(tensor1);
  CHECK(FloatEq(norm_val, 5.0f));  // sqrt(9+16) = 5
  std::cout << "[Pass] norm() L2 norm.\n";

  // Test L1 norm
  Tensor<float> tensor2(2, 2);
  std::vector<float> values2 = {-3, 4, -5, 6};
  tensor2.Fill(values2, false);
  float norm1_val = norm1(tensor2);
  CHECK(FloatEq(norm1_val, 18.0f));  // |−3|+|4|+|−5|+|6| = 18
  std::cout << "[Pass] norm1() L1 norm.\n";

  // Test dot product
  Tensor<float> vec1(1, 3);
  Tensor<float> vec2(1, 3);
  std::vector<float> v1_values = {1, 2, 3};
  std::vector<float> v2_values = {4, 5, 6};
  vec1.Fill(v1_values, false);
  vec2.Fill(v2_values, false);
  float dot_val = dot(vec1, vec2);
  CHECK(FloatEq(dot_val, 32.0f));  // 1*4 + 2*5 + 3*6 = 32
  std::cout << "[Pass] dot() product.\n\n";
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
  TestTensorAdd();
  TestTensorSubtract();
  TestTensorMultiply();
  TestTensorDivide();
  TestScalarAdd();
  TestScalarSub();
  TestScalarMul();
  TestScalarDiv();
  TestElementExp();
  TestElementClip();
  TestBatchDeterminant();
  TestBatchTrace();
  TestBatchInverse();
  TestBatchTranspose();
  TestNormsAndDot();

  std::cout << "====== All test cases passed! ======\n";
  return 0;
}
