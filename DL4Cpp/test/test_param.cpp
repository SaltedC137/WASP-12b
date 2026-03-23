/**
 * @file test_param.cpp
 * @brief Tests for RuntimeParameter class hierarchy
 * @date 2026-03-23
 */

#include "runtime/rt_param.hpp"
#include "runtime/rt_type.hpp"
#include "utils/check.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>

using namespace ctl;

// Helper function to compare floats with tolerance
bool FloatEq(float a, float b, float eps = 1e-5f) {
  return std::abs(a - b) < eps;
}

// ===== RuntimeParameterInt Tests =====

void TestParameterInt() {
  std::cout << "--- Test 1: RuntimeParameterInt ---\n";
  
  // Default constructor
  RuntimeParameterInt param1;
  CHECK(param1.type == RuntimeParameterType::ParameterInt);
  CHECK_EQ(param1.value, 0);
  std::cout << "[Pass] Default constructor.\n";
  
  // Parameterized constructor
  RuntimeParameterInt param2(42);
  CHECK(param2.type == RuntimeParameterType::ParameterInt);
  CHECK_EQ(param2.value, 42);
  std::cout << "[Pass] Parameterized constructor.\n";
  
  // Negative value
  RuntimeParameterInt param3(-100);
  CHECK_EQ(param3.value, -100);
  std::cout << "[Pass] Negative value.\n\n";
}

// ===== RuntimeParameterFloat Tests =====

void TestParameterFloat() {
  std::cout << "--- Test 2: RuntimeParameterFloat ---\n";
  
  // Default constructor
  RuntimeParameterFloat param1;
  CHECK(param1.type == RuntimeParameterType::ParameterFloat);
  CHECK(FloatEq(param1.value, 0.0f));
  std::cout << "[Pass] Default constructor.\n";
  
  // Parameterized constructor
  RuntimeParameterFloat param2(3.14f);
  CHECK(param2.type == RuntimeParameterType::ParameterFloat);
  CHECK(FloatEq(param2.value, 3.14f));
  std::cout << "[Pass] Parameterized constructor.\n";
  
  // Negative value
  RuntimeParameterFloat param3(-2.5f);
  CHECK(FloatEq(param3.value, -2.5f));
  std::cout << "[Pass] Negative value.\n";
  
  // Small value
  RuntimeParameterFloat param4(1e-6f);
  CHECK(FloatEq(param4.value, 1e-6f));
  std::cout << "[Pass] Small value.\n\n";
}

// ===== RuntimeParameterString Tests =====

void TestParameterString() {
  std::cout << "--- Test 3: RuntimeParameterString ---\n";
  
  // Default constructor
  RuntimeParameterString param1;
  CHECK(param1.type == RuntimeParameterType::ParameterString);
  CHECK(param1.value.empty());
  std::cout << "[Pass] Default constructor.\n";
  
  // Parameterized constructor
  RuntimeParameterString param2("relu");
  CHECK(param2.type == RuntimeParameterType::ParameterString);
  CHECK_EQ(param2.value, "relu");
  std::cout << "[Pass] Parameterized constructor.\n";
  
  // Empty string
  RuntimeParameterString param3(std::string(""));
  CHECK(param3.value.empty());
  std::cout << "[Pass] Empty string.\n";
  
  // Long string
  RuntimeParameterString param4("this_is_a_very_long_activation_function_name_for_testing");
  CHECK(!param4.value.empty());
  std::cout << "[Pass] Long string.\n\n";
}

// ===== RuntimeParameterBool Tests =====

void TestParameterBool() {
  std::cout << "--- Test 4: RuntimeParameterBool ---\n";
  
  // Default constructor
  RuntimeParameterBool param1;
  CHECK(param1.type == RuntimeParameterType::ParameterBool);
  CHECK_EQ(param1.value, false);
  std::cout << "[Pass] Default constructor.\n";
  
  // Parameterized constructor (true)
  RuntimeParameterBool param2(true);
  CHECK(param2.type == RuntimeParameterType::ParameterBool);
  CHECK_EQ(param2.value, true);
  std::cout << "[Pass] Parameterized constructor (true).\n";
  
  // Parameterized constructor (false)
  RuntimeParameterBool param3(false);
  CHECK_EQ(param3.value, false);
  std::cout << "[Pass] Parameterized constructor (false).\n\n";
}

// ===== RuntimeParameterIntArray Tests =====

void TestParameterIntArray() {
  std::cout << "--- Test 5: RuntimeParameterIntArray ---\n";
  
  // Default constructor
  RuntimeParameterIntArray param1;
  CHECK(param1.type == RuntimeParameterType::ParameterIntArray);
  CHECK(param1.value.empty());
  std::cout << "[Pass] Default constructor.\n";
  
  // Parameterized constructor
  std::vector<int32_t> values = {1, 2, 3, 4};
  RuntimeParameterIntArray param2(values);
  CHECK(param2.type == RuntimeParameterType::ParameterIntArray);
  CHECK_EQ(param2.value.size(), 4u);
  CHECK_EQ(param2.value[0], 1);
  CHECK_EQ(param2.value[3], 4);
  std::cout << "[Pass] Parameterized constructor.\n";
  
  // Single element
  std::vector<int32_t> single = {42};
  RuntimeParameterIntArray param3(single);
  CHECK_EQ(param3.value.size(), 1u);
  CHECK_EQ(param3.value[0], 42);
  std::cout << "[Pass] Single element.\n";
  
  // Negative values
  std::vector<int32_t> negs = {-1, -2, -3};
  RuntimeParameterIntArray param4(negs);
  CHECK_EQ(param4.value[0], -1);
  CHECK_EQ(param4.value[2], -3);
  std::cout << "[Pass] Negative values.\n\n";
}

// ===== RuntimeParameterFloatArray Tests =====

void TestParameterFloatArray() {
  std::cout << "--- Test 6: RuntimeParameterFloatArray ---\n";
  
  // Default constructor
  RuntimeParameterFloatArray param1;
  CHECK(param1.type == RuntimeParameterType::ParameterFloatArray);
  CHECK(param1.value.empty());
  std::cout << "[Pass] Default constructor.\n";
  
  // Parameterized constructor
  std::vector<float> values = {1.5f, 2.5f, 3.5f};
  RuntimeParameterFloatArray param2(values);
  CHECK(param2.type == RuntimeParameterType::ParameterFloatArray);
  CHECK_EQ(param2.value.size(), 3u);
  CHECK(FloatEq(param2.value[0], 1.5f));
  CHECK(FloatEq(param2.value[2], 3.5f));
  std::cout << "[Pass] Parameterized constructor.\n";
  
  // Single element
  std::vector<float> single = {3.14f};
  RuntimeParameterFloatArray param3(single);
  CHECK_EQ(param3.value.size(), 1u);
  CHECK(FloatEq(param3.value[0], 3.14f));
  std::cout << "[Pass] Single element.\n";
  
  // Mixed positive/negative
  std::vector<float> mixed = {-1.0f, 0.0f, 1.0f};
  RuntimeParameterFloatArray param4(mixed);
  CHECK(FloatEq(param4.value[0], -1.0f));
  CHECK(FloatEq(param4.value[1], 0.0f));
  CHECK(FloatEq(param4.value[2], 1.0f));
  std::cout << "[Pass] Mixed positive/negative.\n\n";
}

// ===== RuntimeParameterStringArray Tests =====

void TestParameterStringArray() {
  std::cout << "--- Test 7: RuntimeParameterStringArray ---\n";
  
  // Default constructor
  RuntimeParameterStringArray param1;
  CHECK(param1.type == RuntimeParameterType::ParameterStringArray);
  CHECK(param1.value.empty());
  std::cout << "[Pass] Default constructor.\n";
  
  // Parameterized constructor
  std::vector<std::string> values = {"hello", "world"};
  RuntimeParameterStringArray param2(values);
  CHECK(param2.type == RuntimeParameterType::ParameterStringArray);
  CHECK_EQ(param2.value.size(), 2u);
  CHECK_EQ(param2.value[0], "hello");
  CHECK_EQ(param2.value[1], "world");
  std::cout << "[Pass] Parameterized constructor.\n";
  
  // Empty strings
  std::vector<std::string> empty_vals = {"", ""};
  RuntimeParameterStringArray param3(empty_vals);
  CHECK_EQ(param3.value.size(), 2u);
  CHECK(param3.value[0].empty());
  std::cout << "[Pass] Empty strings.\n\n";
}

// ===== Polymorphism Tests =====

void TestParameterPolymorphism() {
  std::cout << "--- Test 8: Polymorphism (Base Class Pointer) ---\n";
  
  // Store different parameter types in base class pointers
  std::vector<std::unique_ptr<RuntimeParameter>> params;
  
  params.push_back(std::make_unique<RuntimeParameterInt>(100));
  params.push_back(std::make_unique<RuntimeParameterFloat>(3.14f));
  params.push_back(std::make_unique<RuntimeParameterString>("test"));
  params.push_back(std::make_unique<RuntimeParameterBool>(true));
  
  CHECK(params[0]->type == RuntimeParameterType::ParameterInt);
  CHECK(params[1]->type == RuntimeParameterType::ParameterFloat);
  CHECK(params[2]->type == RuntimeParameterType::ParameterString);
  CHECK(params[3]->type == RuntimeParameterType::ParameterBool);
  
  // Downcast and verify values
  auto* int_param = dynamic_cast<RuntimeParameterInt*>(params[0].get());
  CHECK(int_param != nullptr);
  CHECK_EQ(int_param->value, 100);
  
  auto* float_param = dynamic_cast<RuntimeParameterFloat*>(params[1].get());
  CHECK(float_param != nullptr);
  CHECK(FloatEq(float_param->value, 3.14f));
  
  std::cout << "[Pass] Polymorphism and downcasting.\n\n";
}

// ===== Use Case: Conv2D Parameters =====


int main() {
  std::cout << "========================================\n";
  std::cout << "  RuntimeParameter Test Suite\n";
  std::cout << "========================================\n\n";
  
  TestParameterInt();
  TestParameterFloat();
  TestParameterString();
  TestParameterBool();
  TestParameterIntArray();
  TestParameterFloatArray();
  TestParameterStringArray();
  TestParameterPolymorphism();
  
  std::cout << "========================================\n";
  std::cout << "  All Parameter Tests Completed\n";
  std::cout << "========================================\n";
  
  return 0;
}
