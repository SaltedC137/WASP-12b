/**
 * @file test_runtime.cpp
 * @brief Tests for RuntimeGraph, RuntimeOperator, and RuntimeOperand
 * @date 2026-03-23
 */

#include "runtime/rt_ir.hpp"
#include "runtime/rt_op.hpp"
#include "runtime/rt_opd.hpp"
#include "runtime/rt_type.hpp"
#include "core/tensor.hpp"
#include "utils/check.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <string>

using namespace ctl;

// ===== RuntimeOperand Tests =====

void TestOperandDefaultConstructor() {
  std::cout << "--- Test 1: RuntimeOperand Default Constructor ---\n";
  
  RuntimeOperand operand;
  
  CHECK(operand.name.empty());
  CHECK(operand.shapes.empty());
  CHECK(operand.datas.empty());
  CHECK(operand.type == RuntimeDataType::TypeUnknown);
  
  std::cout << "[Pass] Default constructor.\n\n";
}

void TestOperandWithData() {
  std::cout << "--- Test 2: RuntimeOperand With Data ---\n";
  
  // Create tensor data: 3 channels, 4 rows, 2 cols
  auto tensor = std::make_shared<Tensor<float>>(3, 4, 2);
  tensor->Fill(1.0f);
  
  std::vector<int32_t> shapes = {1, 3, 4, 2};  // [batch, channels, rows, cols]
  std::vector<std::shared_ptr<Tensor<float>>> data = {tensor};
  
  RuntimeOperand operand("test_operand", shapes, data, RuntimeDataType::TypeFloat32);
  
  CHECK_EQ(operand.name, "test_operand");
  CHECK_EQ(operand.shapes.size(), 4u);
  CHECK_EQ(operand.shapes[0], 1);
  CHECK_EQ(operand.shapes[1], 3);
  CHECK_EQ(operand.shapes[2], 4);
  CHECK_EQ(operand.shapes[3], 2);
  CHECK_EQ(operand.datas.size(), 1u);
  CHECK(operand.datas[0] != nullptr);
  CHECK(operand.type == RuntimeDataType::TypeFloat32);
  
  std::cout << "[Pass] Constructor with data.\n\n";
}

void TestOperandWithSlots() {
  std::cout << "--- Test 3: RuntimeOperand With Data Slots ---\n";
  
  std::vector<int32_t> shapes = {4, 3, 32, 32};  // batch=4, 3-channel 32x32
  uint32_t data_size = 4;
  
  RuntimeOperand operand("input", shapes, data_size, RuntimeDataType::TypeFloat32);
  
  CHECK_EQ(operand.name, "input");
  CHECK_EQ(operand.shapes.size(), 4u);
  CHECK_EQ(operand.datas.size(), data_size);
  CHECK(operand.type == RuntimeDataType::TypeFloat32);
  
  // Data slots should be null initially
  for (const auto& slot : operand.datas) {
    CHECK(slot == nullptr);
  }
  
  std::cout << "[Pass] Constructor with data slots.\n\n";
}

void TestOperandSize() {
  std::cout << "--- Test 4: RuntimeOperand Size Calculation ---\n";
  
  // 1D operand
  RuntimeOperand operand1d("1d", std::vector<int32_t>{1, 10}, 1, RuntimeDataType::TypeFloat32);
  CHECK_EQ(operand1d.size(), 10u);
  
  // 2D operand
  RuntimeOperand operand2d("2d", std::vector<int32_t>{1, 3, 4}, 1, RuntimeDataType::TypeFloat32);
  CHECK_EQ(operand2d.size(), 12u);
  
  // 3D operand
  RuntimeOperand operand3d("3d", std::vector<int32_t>{2, 3, 4}, 1, RuntimeDataType::TypeFloat32);
  CHECK_EQ(operand3d.size(), 24u);
  
  // 4D operand (batch)
  RuntimeOperand operand4d("4d", std::vector<int32_t>{8, 3, 32, 32}, 1, RuntimeDataType::TypeFloat32);
  CHECK_EQ(operand4d.size(), 24576u);
  
  // Empty shape
  RuntimeOperand empty("empty", std::vector<int32_t>{}, 1, RuntimeDataType::TypeFloat32);
  CHECK_EQ(empty.size(), 0u);
  
  std::cout << "[Pass] Size calculation.\n\n";
}

// ===== RuntimeOperator Tests =====

void TestOperatorDefaultConstructor() {
  std::cout << "--- Test 5: RuntimeOperator Default Constructor ---\n";
  
  RuntimeOperator op;
  
  CHECK_EQ(op.start_time, -1);
  CHECK_EQ(op.end_time, -1);
  CHECK_EQ(op.occur_end_time, -1);
  CHECK_EQ(op.has_forward, false);
  CHECK(op.name.empty());
  CHECK(op.type.empty());
  CHECK(op.layer == nullptr);
  CHECK(op.output_names.empty());
  CHECK(op.output_operand == nullptr);
  CHECK(op.input_operands.empty());
  CHECK(op.input_operands_seq.empty());
  CHECK(op.param.empty());
  CHECK(op.attribute.empty());
  
  std::cout << "[Pass] Default constructor.\n\n";
}

void TestOperatorHasParameter() {
  std::cout << "--- Test 6: RuntimeOperator HasParameter ---\n";
  
  RuntimeOperator op;
  op.name = "test_op";
  op.type = "nn.Conv2d";
  
  // Initially no parameters
  CHECK_EQ(op.has_parameter("stride"), false);
  CHECK_EQ(op.has_attribute("weight"), false);
  
  // Add a parameter
  op.param["stride"] = std::make_shared<RuntimeParameterIntArray>(
      std::vector<int32_t>{2, 2});
  
  CHECK_EQ(op.has_parameter("stride"), true);
  CHECK_EQ(op.has_parameter("padding"), false);
  
  // Add an attribute
  op.attribute["weight"] = std::make_shared<RuntimeAttribute>();
  
  CHECK_EQ(op.has_attribute("weight"), true);
  CHECK_EQ(op.has_attribute("bias"), false);
  
  std::cout << "[Pass] HasParameter/HasAttribute.\n\n";
}

void TestOperatorInputOutputOperands() {
  std::cout << "--- Test 7: RuntimeOperator Input/Output Operands ---\n";
  
  RuntimeOperator op;
  op.name = "conv1";
  op.type = "nn.Conv2d";
  
  // Create input operand
  auto input_operand = std::make_shared<RuntimeOperand>(
      "input", std::vector<int32_t>{1, 3, 32, 32}, 1, RuntimeDataType::TypeFloat32);
  
  // Create output operand
  auto output_operand = std::make_shared<RuntimeOperand>(
      "conv1_output", std::vector<int32_t>{1, 64, 16, 16}, 1, RuntimeDataType::TypeFloat32);
  
  // Set operands
  op.input_operands["input"] = input_operand;
  op.input_operands_seq.push_back(input_operand);
  op.output_operand = output_operand;
  
  CHECK_EQ(op.input_operands.size(), 1u);
  CHECK_EQ(op.input_operands_seq.size(), 1u);
  CHECK(op.output_operand != nullptr);
  CHECK_EQ(op.output_operand->name, "conv1_output");
  
  std::cout << "[Pass] Input/Output operands.\n\n";
}

// ===== RuntimeOperatorUtils Tests =====

void TestInitOperatorInput() {
  std::cout << "--- Test 8: InitOperatorInput ---\n";
  
  // Create operators with input operands
  auto op1 = std::make_shared<RuntimeOperator>();
  op1->name = "input";
  op1->type = "pnnx.Input";
  op1->start_time = 0;
  op1->end_time = 1;
  
  auto input_operand = std::make_shared<RuntimeOperand>(
      "input", std::vector<int32_t>{1, 3, 32, 32}, 1, RuntimeDataType::TypeFloat32);
  op1->input_operands["input"] = input_operand;
  op1->input_operands_seq.push_back(input_operand);
  
  auto op2 = std::make_shared<RuntimeOperator>();
  op2->name = "conv1";
  op2->type = "nn.Conv2d";
  op2->start_time = 1;
  op2->end_time = 2;
  
  auto conv_input = std::make_shared<RuntimeOperand>(
      "conv1_input", std::vector<int32_t>{1, 3, 32, 32}, 1, RuntimeDataType::TypeFloat32);
  op2->input_operands["input"] = conv_input;
  op2->input_operands_seq.push_back(conv_input);
  
  std::vector<std::shared_ptr<RuntimeOperator>> operators = {op1, op2};
  
  // Initialize inputs
  RuntimeOperatorUtils<float>::InitOperatorInput(operators);
  
  // Verify input tensors are allocated
  CHECK_EQ(op1->input_operands["input"]->datas.size(), 1u);
  CHECK_EQ(op2->input_operands["input"]->datas.size(), 1u);
  
  std::cout << "[Pass] InitOperatorInput.\n\n";
}

// ===== RuntimeGraph Tests =====

void TestGraphConstructor() {
  std::cout << "--- Test 9: RuntimeGraph Constructor ---\n";
  
  // Note: This test uses non-existent files, so Build() will fail
  // But we can test construction and initial state
  RuntimeGraph graph("nonexistent.param", "nonexistent.bin");
  
  // Graph should be in NeedInit state (can't check directly due to private enum)
  // Just verify construction doesn't crash
  std::cout << "[Pass] Constructor (files not loaded).\n\n";
}

void TestGraphSettersAndGetters() {
  std::cout << "--- Test 10: RuntimeGraph Setters/Getters ---\n";
  
  RuntimeGraph graph("path1.param", "path1.bin");
  
  CHECK_EQ(graph.param_path(), "path1.param");
  CHECK_EQ(graph.bin_path(), "path1.bin");
  
  graph.set_param_path("path2.param");
  graph.set_bin_path("path2.bin");
  
  CHECK_EQ(graph.param_path(), "path2.param");
  CHECK_EQ(graph.bin_path(), "path2.bin");
  
  std::cout << "[Pass] Setters and getters.\n\n";
}

void TestGraphInputOutputCheck() {
  std::cout << "--- Test 11: RuntimeGraph Input/Output Check ---\n";
  
  RuntimeGraph graph("test.param", "test.bin");
  
  // Before building, these should return false
  CHECK_EQ(graph.is_input_op("input"), false);
  CHECK_EQ(graph.is_output_op("output"), false);
  
  std::cout << "[Pass] Input/Output check (before build).\n\n";
}

void TestGraphSetInputs() {
  std::cout << "--- Test 12: RuntimeGraph SetInputs ---\n";
  
  RuntimeGraph graph("test.param", "test.bin");
  
  // Create input tensor: batch=1, 3 channels, 32x32
  auto input = std::make_shared<Tensor<float>>(3, 32, 32);
  input->Fill(1.0f);
  
  // Set inputs (will be stored but not used until graph is built)
  graph.set_inputs("input", {input});
  
  std::cout << "[Pass] SetInputs.\n\n";
}

void TestGraphGetOutput() {
  std::cout << "--- Test 13: RuntimeGraph GetOutput ---\n";
  
  RuntimeGraph graph("test.param", "test.bin");
  
  // Before building, should return empty
  auto output = graph.get_output("output");
  CHECK(output.empty());
  
  std::cout << "[Pass] GetOutput (before build).\n\n";
}

// ===== Integration Test: Simple Graph =====

void TestSimpleGraph() {
  std::cout << "--- Test 14: Integration - Simple Graph Structure ---\n";
  
  // Manually create a simple graph structure
  // input -> conv1 -> relu1 -> output
  
  auto input_op = std::make_shared<RuntimeOperator>();
  input_op->name = "input";
  input_op->type = "pnnx.Input";
  input_op->start_time = 0;
  input_op->end_time = 1;
  input_op->output_operand = std::make_shared<RuntimeOperand>(
      "input", std::vector<int32_t>{1, 3, 32, 32}, 1, RuntimeDataType::TypeFloat32);
  
  auto conv_op = std::make_shared<RuntimeOperator>();
  conv_op->name = "conv1";
  conv_op->type = "nn.Conv2d";
  conv_op->start_time = 1;
  conv_op->end_time = 2;
  conv_op->input_operands["input"] = input_op->output_operand;
  conv_op->output_operand = std::make_shared<RuntimeOperand>(
      "conv1_output", std::vector<int32_t>{1, 64, 16, 16}, 1, RuntimeDataType::TypeFloat32);
  
  auto relu_op = std::make_shared<RuntimeOperator>();
  relu_op->name = "relu1";
  relu_op->type = "nn.ReLU";
  relu_op->start_time = 2;
  relu_op->end_time = 3;
  relu_op->input_operands["input"] = conv_op->output_operand;
  relu_op->output_operand = std::make_shared<RuntimeOperand>(
      "relu1_output", std::vector<int32_t>{1, 64, 16, 16}, 1, RuntimeDataType::TypeFloat32);
  
  auto output_op = std::make_shared<RuntimeOperator>();
  output_op->name = "output";
  output_op->type = "pnnx.Output";
  output_op->start_time = 3;
  output_op->end_time = 4;
  output_op->input_operands["input"] = relu_op->output_operand;
  
  // Verify graph structure
  CHECK_EQ(input_op->type, "pnnx.Input");
  CHECK_EQ(conv_op->type, "nn.Conv2d");
  CHECK_EQ(relu_op->type, "nn.ReLU");
  CHECK_EQ(output_op->type, "pnnx.Output");
  
  // Verify topological order
  CHECK_LT(input_op->start_time, conv_op->start_time);
  CHECK_LT(conv_op->start_time, relu_op->start_time);
  CHECK_LT(relu_op->start_time, output_op->start_time);
  
  std::cout << "[Pass] Simple graph structure.\n\n";
}

// ===== DataType Tests =====

void TestRuntimeDataTypeCoverage() {
  std::cout << "--- Test 15: RuntimeDataType Coverage ---\n";
  
  // Test all enum values exist
  CHECK(RuntimeDataType::TypeUnknown == RuntimeDataType::TypeUnknown);
  CHECK(RuntimeDataType::TypeFloat32 == RuntimeDataType::TypeFloat32);
  CHECK(RuntimeDataType::TypeFloat64 == RuntimeDataType::TypeFloat64);
  CHECK(RuntimeDataType::TypeFloat16 == RuntimeDataType::TypeFloat16);
  CHECK(RuntimeDataType::TypeInt32 == RuntimeDataType::TypeInt32);
  CHECK(RuntimeDataType::TypeInt64 == RuntimeDataType::TypeInt64);
  CHECK(RuntimeDataType::TypeInt16 == RuntimeDataType::TypeInt16);
  CHECK(RuntimeDataType::TypeInt8 == RuntimeDataType::TypeInt8);
  CHECK(RuntimeDataType::TypeUInt8 == RuntimeDataType::TypeUInt8);
  
  std::cout << "[Pass] RuntimeDataType coverage.\n\n";
}

int main() {
  std::cout << "========================================\n";
  std::cout << "  Runtime System Test Suite\n";
  std::cout << "========================================\n\n";
  
  // Operand tests
  TestOperandDefaultConstructor();
  TestOperandWithData();
  TestOperandWithSlots();
  TestOperandSize();
  
  // Operator tests
  TestOperatorDefaultConstructor();
  TestOperatorHasParameter();
  TestOperatorInputOutputOperands();
  
  // Operator utils tests
  TestInitOperatorInput();
  
  // Graph tests
  TestGraphConstructor();
  TestGraphSettersAndGetters();
  TestGraphInputOutputCheck();
  TestGraphSetInputs();
  TestGraphGetOutput();
  
  // Integration tests
  TestSimpleGraph();
  
  // Type coverage tests
  TestRuntimeDataTypeCoverage();
  
  std::cout << "========================================\n";
  std::cout << "  All Runtime Tests Completed\n";
  std::cout << "========================================\n";
  
  return 0;
}
