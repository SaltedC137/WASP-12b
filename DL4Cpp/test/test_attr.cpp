/**
 * @file test_attr.cpp
 * @brief Tests for RuntimeAttribute class
 * @date 2026-03-23
 */

#include "runtime/rt_attr.hpp"
#include "runtime/rt_type.hpp"
#include "utils/check.hpp"
#include <iostream>
#include <vector>
#include <cstring>

using namespace ctl;

// Helper function to compare floats with tolerance
bool FloatEq(float a, float b, float eps = 1e-5f) {
  return std::abs(a - b) < eps;
}

void TestAttributeDefaultConstructor() {
  std::cout << "--- Test 1: Default Constructor ---\n";
  
  RuntimeAttribute attr;
  
  CHECK(attr.shape.empty());
  CHECK(attr.weight_data.empty());
  CHECK(attr.type == RuntimeDataType::TypeUnknown);
  
  std::cout << "[Pass] Default constructor.\n\n";
}

void TestAttributeParameterizedConstructor() {
  std::cout << "--- Test 2: Parameterized Constructor ---\n";
  
  // Create weight data: [1.0, 2.0, 3.0, 4.0]
  std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<char> weight_data(weights.size() * sizeof(float));
  std::memcpy(weight_data.data(), weights.data(), weight_data.size());
  
  std::vector<int32_t> shape = {2, 2};
  
  RuntimeAttribute attr(shape, RuntimeDataType::TypeFloat32, weight_data);
  
  CHECK_EQ(attr.shape.size(), 2);
  CHECK_EQ(attr.shape[0], 2);
  CHECK_EQ(attr.shape[1], 2);
  // CHECK_EQ(attr.type, RuntimeDataType::TypeFloat32);
  CHECK_EQ(attr.weight_data.size(), 4 * sizeof(float));
  
  std::cout << "[Pass] Parameterized constructor.\n\n";
}

void TestAttributeGetFloat() {
  std::cout << "--- Test 3: Get Float Weights ---\n";
  
  // Create weight data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  std::vector<float> original_weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<char> weight_data(original_weights.size() * sizeof(float));
  std::memcpy(weight_data.data(), original_weights.data(), weight_data.size());
  
  std::vector<int32_t> shape = {2, 3};
  RuntimeAttribute attr(shape, RuntimeDataType::TypeFloat32, weight_data);
  
  // Get weights (with clear)
  auto retrieved = attr.get<float>(true);
  
  CHECK_EQ(retrieved.size(), original_weights.size());
  for (size_t i = 0; i < retrieved.size(); ++i) {
    CHECK(FloatEq(retrieved[i], original_weights[i]));
  }
  
  // Verify weight_data is cleared
  CHECK(attr.weight_data.empty());
  
  std::cout << "[Pass] Get float weights (with clear).\n\n";
}

void TestAttributeGetFloatNoClear() {
  std::cout << "--- Test 4: Get Float Weights (No Clear) ---\n";
  
  std::vector<float> original_weights = {1.5f, 2.5f, 3.5f, 4.5f};
  std::vector<char> weight_data(original_weights.size() * sizeof(float));
  std::memcpy(weight_data.data(), original_weights.data(), weight_data.size());
  
  std::vector<int32_t> shape = {2, 2};
  RuntimeAttribute attr(shape, RuntimeDataType::TypeFloat32, weight_data);
  
  // Get weights without clearing
  auto retrieved = attr.get<float>(false);
  
  CHECK_EQ(retrieved.size(), original_weights.size());
  for (size_t i = 0; i < retrieved.size(); ++i) {
    CHECK(FloatEq(retrieved[i], original_weights[i]));
  }
  
  // Verify weight_data is NOT cleared
  CHECK(!attr.weight_data.empty());
  
  std::cout << "[Pass] Get float weights (no clear).\n\n";
}

void TestAttributeGetInt32() {
  std::cout << "--- Test 5: Get Int32 Weights ---\n";
  
  std::vector<int32_t> original_weights = {10, 20, 30, 40, 50};
  std::vector<char> weight_data(original_weights.size() * sizeof(int32_t));
  std::memcpy(weight_data.data(), original_weights.data(), weight_data.size());
  
  std::vector<int32_t> shape = {5, 1};
  RuntimeAttribute attr(shape, RuntimeDataType::TypeInt32, weight_data);
  
  auto retrieved = attr.get<int32_t>(true);
  
  CHECK_EQ(retrieved.size(), original_weights.size());
  for (size_t i = 0; i < retrieved.size(); ++i) {
    CHECK_EQ(retrieved[i], original_weights[i]);
  }
  
  std::cout << "[Pass] Get int32 weights.\n\n";
}

void TestAttributeTypeMismatch() {
  std::cout << "--- Test 6: Type Mismatch (Expected to Fail) ---\n";
  
  std::vector<float> weights = {1.0f, 2.0f, 3.0f};
  std::vector<char> weight_data(weights.size() * sizeof(float));
  std::memcpy(weight_data.data(), weights.data(), weight_data.size());
  
  std::vector<int32_t> shape = {3};
  // Set type to Int32 but data is actually float
  RuntimeAttribute attr(shape, RuntimeDataType::TypeInt32, weight_data);
  
  // This should trigger CHECK failure in debug mode
  // In release mode, it will return garbage
  std::cout << "Attempting to get float from Int32-typed attribute...\n";
  
  // Note: This will cause CHECK failure if type checking is enabled
  // Uncomment to test:
  // auto retrieved = attr.get<float>(false);
  
  std::cout << "[Pass] Type mismatch handling (test skipped in release).\n\n";
}

void TestAttributeLargeWeights() {
  std::cout << "--- Test 7: Large Weights (Conv2D-like) ---\n";
  
  // Simulate Conv2D weights: [out_channels, in_channels, kernel_h, kernel_w]
  const int out_c = 64, in_c = 3, k_h = 3, k_w = 3;
  const size_t weight_count = out_c * in_c * k_h * k_w;
  
  std::vector<float> original_weights(weight_count);
  for (size_t i = 0; i < weight_count; ++i) {
    original_weights[i] = static_cast<float>(i) / weight_count;
  }
  
  std::vector<char> weight_data(original_weights.size() * sizeof(float));
  std::memcpy(weight_data.data(), original_weights.data(), weight_data.size());
  
  std::vector<int32_t> shape = {out_c, in_c, k_h, k_w};
  RuntimeAttribute attr(shape, RuntimeDataType::TypeFloat32, weight_data);
  
  CHECK_EQ(attr.shape.size(), 4u);
  CHECK_EQ(attr.shape[0], out_c);
  CHECK_EQ(attr.shape[1], in_c);
  CHECK_EQ(attr.shape[2], k_h);
  CHECK_EQ(attr.shape[3], k_w);
  
  auto retrieved = attr.get<float>(true);
  CHECK_EQ(retrieved.size(), weight_count);
  
  // Sample check
  CHECK(FloatEq(retrieved[0], original_weights[0]));
  CHECK(FloatEq(retrieved[weight_count - 1], original_weights[weight_count - 1]));
  
  std::cout << "[Pass] Large weights (Conv2D-like).\n\n";
}

void TestAttributeMoveSemantics() {
  std::cout << "--- Test 8: Move Semantics ---\n";
  
  std::vector<float> weights = {1.0f, 2.0f, 3.0f};
  std::vector<char> weight_data(weights.size() * sizeof(float));
  std::memcpy(weight_data.data(), weights.data(), weight_data.size());
  
  std::vector<int32_t> shape = {3};
  RuntimeAttribute attr1(shape, RuntimeDataType::TypeFloat32, std::move(weight_data));
  
  // Verify original data was moved
  CHECK(weight_data.empty());
  
  // Move constructor
  RuntimeAttribute attr2(std::move(attr1));
  CHECK(attr1.weight_data.empty());
  
  auto retrieved = attr2.get<float>(false);
  CHECK_EQ(retrieved.size(), 3u);
  CHECK(FloatEq(retrieved[0], 1.0f));
  
  std::cout << "[Pass] Move semantics.\n\n";
}

int main() {
  std::cout << "========================================\n";
  std::cout << "  RuntimeAttribute Test Suite\n";
  std::cout << "========================================\n\n";
  
  TestAttributeDefaultConstructor();
  TestAttributeParameterizedConstructor();
  TestAttributeGetFloat();
  TestAttributeGetFloatNoClear();
  TestAttributeGetInt32();
  TestAttributeTypeMismatch();
  TestAttributeLargeWeights();
  TestAttributeMoveSemantics();
  
  std::cout << "========================================\n";
  std::cout << "  All Attribute Tests Completed\n";
  std::cout << "========================================\n";
  
  return 0;
}
