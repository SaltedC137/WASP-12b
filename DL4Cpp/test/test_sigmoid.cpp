/**
 * @file test_sigmoid.cpp
 * @brief Brute-force tests for Sigmoid activation function
 * @date 2026-03-24
 */

#include "core/tensor.hpp"
#include "nn/ops/sigmoid.hpp"
#include "nn/ops/activation.hpp"
#include "utils/check.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace ctl;
using namespace ctl::nn;

// Reference sigmoid implementation
float ReferenceSigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

// Helper function to compare floats with tolerance
bool FloatEq(float a, float b, float eps = 1e-5f) {
  return std::abs(a - b) < eps;
}

void TestSigmoid_Nullptr() {
  std::cout << "--- Test 1: Sigmoid with nullptr (should assert) ---\n";
  // Skip actual test as it would crash
  std::cout << "[Skip] nullptr test (would trigger CHECK macro)\n\n";
}

void TestSigmoid_EmptyTensor() {
  std::cout << "--- Test 2: Sigmoid with empty tensor ---\n";
  // Skip actual test as it would crash
  std::cout << "[Skip] empty tensor test (would trigger CHECK macro)\n\n";
}

void TestSigmoid_SingleElement() {
  std::cout << "--- Test 3: Sigmoid single element ---\n";

  auto input = std::make_shared<Tensor<float>>(1, 1, 1);
  auto output = std::make_shared<Tensor<float>>(1, 1, 1);

  std::vector<float> values = {0.0f};
  input->Fill(values, false);

  auto activation_func = ApplySSEActivation(ActivationType::ActivationSigmoid);
  activation_func(input, output);

  float expected = ReferenceSigmoid(0.0f); // 0.5
  CHECK(FloatEq(output->at(0, 0, 0), expected));
  CHECK(FloatEq(output->at(0, 0, 0), 0.5f));
  std::cout << "[Pass] Sigmoid(0) = 0.5\n\n";
}

void TestSigmoid_PositiveValues() {
  std::cout << "--- Test 4: Sigmoid positive values ---\n";

  auto input = std::make_shared<Tensor<float>>(1, 1, 5);
  auto output = std::make_shared<Tensor<float>>(1, 1, 5);

  std::vector<float> values = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  input->Fill(values, false);

  auto activation_func = ApplySSEActivation(ActivationType::ActivationSigmoid);
  activation_func(input, output);

  for (int i = 0; i < 5; ++i) {
    float expected = ReferenceSigmoid(values[i]);
    float actual = output->index(i);
    CHECK(FloatEq(actual, expected));
    std::cout << "  Sigmoid(" << values[i] << ") = " << actual
              << " (expected: " << expected << ")\n";
  }
  std::cout << "[Pass] Positive values\n\n";
}

void TestSigmoid_NegativeValues() {
  std::cout << "--- Test 5: Sigmoid negative values ---\n";

  auto input = std::make_shared<Tensor<float>>(1, 1, 5);
  auto output = std::make_shared<Tensor<float>>(1, 1, 5);

  std::vector<float> values = {-4.0f, -3.0f, -2.0f, -1.0f, 0.0f};
  input->Fill(values, false);

  auto activation_func = ApplySSEActivation(ActivationType::ActivationSigmoid);
  activation_func(input, output);

  for (int i = 0; i < 5; ++i) {
    float expected = ReferenceSigmoid(values[i]);
    float actual = output->index(i);
    CHECK(FloatEq(actual, expected));
    std::cout << "  Sigmoid(" << values[i] << ") = " << actual
              << " (expected: " << expected << ")\n";
  }
  std::cout << "[Pass] Negative values\n\n";
}

void TestSigmoid_LargeValues() {
  std::cout << "--- Test 6: Sigmoid large magnitude values ---\n";

  auto input = std::make_shared<Tensor<float>>(1, 1, 6);
  auto output = std::make_shared<Tensor<float>>(1, 1, 6);

  // Large positive should approach 1, large negative should approach 0
  std::vector<float> values = {-100.0f, -50.0f, -10.0f, 10.0f, 50.0f, 100.0f};
  input->Fill(values, false);

  auto activation_func = ApplySSEActivation(ActivationType::ActivationSigmoid);
  activation_func(input, output);

  // Check large negative values approach 0
  CHECK(FloatEq(output->index(0), 0.0f, 1e-4f));
  CHECK(FloatEq(output->index(1), 0.0f, 1e-4f));
  CHECK(FloatEq(output->index(2), 0.0f, 1e-4f));

  // Check large positive values approach 1
  CHECK(FloatEq(output->index(3), 1.0f, 1e-4f));
  CHECK(FloatEq(output->index(4), 1.0f, 1e-4f));
  CHECK(FloatEq(output->index(5), 1.0f, 1e-4f));

  std::cout << "  Sigmoid(-100) = " << output->index(0) << " (approaches 0)\n";
  std::cout << "  Sigmoid(-50)  = " << output->index(1) << " (approaches 0)\n";
  std::cout << "  Sigmoid(-10)  = " << output->index(2) << " (approaches 0)\n";
  std::cout << "  Sigmoid(10)   = " << output->index(3) << " (approaches 1)\n";
  std::cout << "  Sigmoid(50)   = " << output->index(4) << " (approaches 1)\n";
  std::cout << "  Sigmoid(100)  = " << output->index(5) << " (approaches 1)\n";
  std::cout << "[Pass] Large magnitude values\n\n";
}

void TestSigmoid_2DTensor() {
  std::cout << "--- Test 7: Sigmoid 2D tensor ---\n";

  auto input = std::make_shared<Tensor<float>>(4, 4);
  auto output = std::make_shared<Tensor<float>>(4, 4);

  std::vector<float> values = {
    -2.0f, -1.0f, 0.0f, 1.0f,
    -1.5f, -0.5f, 0.5f, 1.5f,
    -1.0f, 0.0f, 1.0f, 2.0f,
    -0.5f, 0.5f, 1.5f, 2.5f
  };
  input->Fill(values, false);

  auto activation_func = ApplySSEActivation(ActivationType::ActivationSigmoid);
  activation_func(input, output);

  for (int i = 0; i < 16; ++i) {
    float expected = ReferenceSigmoid(values[i]);
    float actual = output->index(i);
    CHECK(FloatEq(actual, expected));
  }
  std::cout << "[Pass] 2D tensor (4x4)\n\n";
}

void TestSigmoid_3DTensor() {
  std::cout << "--- Test 8: Sigmoid 3D tensor ---\n";

  auto input = std::make_shared<Tensor<float>>(2, 3, 4);
  auto output = std::make_shared<Tensor<float>>(2, 3, 4);

  std::vector<float> values(24);
  for (int i = 0; i < 24; ++i) {
    values[i] = static_cast<float>(i - 12); // Range from -12 to 11
  }
  input->Fill(values, false);

  auto activation_func = ApplySSEActivation(ActivationType::ActivationSigmoid);
  activation_func(input, output);

  for (int i = 0; i < 24; ++i) {
    float expected = ReferenceSigmoid(values[i]);
    float actual = output->index(i);
    CHECK(FloatEq(actual, expected));
  }
  std::cout << "[Pass] 3D tensor (2x3x4)\n\n";
}

void TestSigmoid_RandomValues() {
  std::cout << "--- Test 9: Sigmoid random values ---\n";

  auto input = std::make_shared<Tensor<float>>(8, 8);
  auto output = std::make_shared<Tensor<float>>(8, 8);

  input->Rand(); // Fill with random values in [0, 1)

  auto activation_func = ApplySSEActivation(ActivationType::ActivationSigmoid);
  activation_func(input, output);

  // Verify all outputs are in (0, 1)
  const float* out_ptr = output->raw_ptr();
  for (int i = 0; i < 64; ++i) {
    CHECK(out_ptr[i] > 0.0f && out_ptr[i] < 1.0f);
  }

  // Verify against reference
  const float* in_ptr = input->raw_ptr();
  for (int i = 0; i < 64; ++i) {
    float expected = ReferenceSigmoid(in_ptr[i]);
    CHECK(FloatEq(out_ptr[i], expected));
  }
  std::cout << "[Pass] Random values (8x8), all outputs in (0, 1)\n\n";
}

void TestSigmoid_InPlace() {
  std::cout << "--- Test 10: Sigmoid in-place operation ---\n";

  auto tensor = std::make_shared<Tensor<float>>(2, 2, 2);
  std::vector<float> values = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 1.5f};
  tensor->Fill(values, false);

  // Use same tensor for input and output
  auto activation_func = ApplySSEActivation(ActivationType::ActivationSigmoid);
  activation_func(tensor, tensor);

  for (int i = 0; i < 8; ++i) {
    float expected = ReferenceSigmoid(values[i]);
    float actual = tensor->index(i);
    CHECK(FloatEq(actual, expected));
  }
  std::cout << "[Pass] In-place operation\n\n";
}

void TestSigmoid_MismatchedSize() {
  std::cout << "--- Test 11: Sigmoid with mismatched tensor sizes ---\n";

  auto input = std::make_shared<Tensor<float>>(2, 2);
  auto output = std::make_shared<Tensor<float>>(3, 3);

  // Skip actual test as it would crash
  std::cout << "[Skip] mismatched size test (would trigger CHECK macro)\n\n";
}

void TestSigmoid_BatchProcessing() {
  std::cout << "--- Test 12: Sigmoid batch processing ---\n";

  // Simulate batch of feature vectors
  auto input = std::make_shared<Tensor<float>>(4, 1, 16); // 4 samples, 16 features each
  auto output = std::make_shared<Tensor<float>>(4, 1, 16);

  std::vector<float> values(64);
  for (int i = 0; i < 64; ++i) {
    values[i] = static_cast<float>(i % 21 - 10); // Alternating values from -10 to 10
  }
  input->Fill(values, false);

  auto activation_func = ApplySSEActivation(ActivationType::ActivationSigmoid);
  activation_func(input, output);

  for (int i = 0; i < 64; ++i) {
    float expected = ReferenceSigmoid(values[i]);
    float actual = output->index(i);
    CHECK(FloatEq(actual, expected));
  }
  std::cout << "[Pass] Batch processing (4 samples x 16 features)\n\n";
}

void TestSigmoid_SpecialValues() {
  std::cout << "--- Test 13: Sigmoid special mathematical properties ---\n";

  auto input = std::make_shared<Tensor<float>>(1, 1, 4);
  auto output = std::make_shared<Tensor<float>>(1, 1, 4);

  // Test sigmoid(-x) = 1 - sigmoid(x)
  std::vector<float> values = {-2.0f, -1.0f, 1.0f, 2.0f};
  input->Fill(values, false);

  auto activation_func = ApplySSEActivation(ActivationType::ActivationSigmoid);
  activation_func(input, output);

  // sigmoid(-2) should equal 1 - sigmoid(2)
  float sig_neg2 = output->index(0);
  float sig_neg1 = output->index(1);
  float sig_pos1 = output->index(2);
  float sig_pos2 = output->index(3);

  CHECK(FloatEq(sig_neg2, 1.0f - sig_pos2));
  CHECK(FloatEq(sig_neg1, 1.0f - sig_pos1));

  std::cout << "  sigmoid(-2) = " << sig_neg2 << ", 1 - sigmoid(2) = " << (1.0f - sig_pos2) << "\n";
  std::cout << "  sigmoid(-1) = " << sig_neg1 << ", 1 - sigmoid(1) = " << (1.0f - sig_pos1) << "\n";
  std::cout << "[Pass] Special property: sigmoid(-x) = 1 - sigmoid(x)\n\n";
}

void TestSigmoid_LargeTensor() {
  std::cout << "--- Test 14: Sigmoid large tensor (stress test) ---\n";

  auto input = std::make_shared<Tensor<float>>(32, 32);
  auto output = std::make_shared<Tensor<float>>(32, 32);

  std::vector<float> values(1024);
  for (int i = 0; i < 1024; ++i) {
    values[i] = static_cast<float>((i % 101) - 50) / 10.0f; // Range -5.0 to 5.0
  }
  input->Fill(values, false);

  auto activation_func = ApplySSEActivation(ActivationType::ActivationSigmoid);
  activation_func(input, output);

  int pass_count = 0;
  for (int i = 0; i < 1024; ++i) {
    float expected = ReferenceSigmoid(values[i]);
    float actual = output->index(i);
    if (FloatEq(actual, expected)) {
      pass_count++;
    }
  }

  CHECK_EQ(pass_count, 1024);
  std::cout << "  Tested 1024 elements, all passed\n";
  std::cout << "[Pass] Large tensor (32x32) stress test\n\n";
}

void TestSigmoid_VeryLargeTensor() {
  std::cout << "--- Test 15: Sigmoid very large tensor (128x128 stress test) ---\n";

  auto input = std::make_shared<Tensor<float>>(128, 128);
  auto output = std::make_shared<Tensor<float>>(128, 128);

  std::vector<float> values(16384);
  for (int i = 0; i < 16384; ++i) {
    values[i] = static_cast<float>((i % 201) - 100) / 10.0f; // Range -10.0 to 10.0
  }
  input->Fill(values, false);

  auto activation_func = ApplySSEActivation(ActivationType::ActivationSigmoid);
  activation_func(input, output);

  int pass_count = 0;
  for (int i = 0; i < 16384; ++i) {
    float expected = ReferenceSigmoid(values[i]);
    float actual = output->index(i);
    if (FloatEq(actual, expected)) {
      pass_count++;
    }
  }

  CHECK_EQ(pass_count, 16384);
  std::cout << "  Tested 16384 elements, all passed\n";
  std::cout << "[Pass] Very large tensor (128x128) stress test\n\n";
}

void TestSigmoid_Deep3DTensor() {
  std::cout << "--- Test 15: Sigmoid deep 3D tensor (8x8x8) ---\n";

  auto input = std::make_shared<Tensor<float>>(8, 8, 8);
  auto output = std::make_shared<Tensor<float>>(8, 8, 8);

  std::vector<float> values(512);
  for (int i = 0; i < 512; ++i) {
    values[i] = static_cast<float>((i % 41) - 20) / 5.0f; // Range -4.0 to 4.0
  }
  input->Fill(values, false);

  auto activation_func = ApplySSEActivation(ActivationType::ActivationSigmoid);
  activation_func(input, output);

  int pass_count = 0;
  for (int i = 0; i < 512; ++i) {
    float expected = ReferenceSigmoid(values[i]);
    float actual = output->index(i);
    if (FloatEq(actual, expected)) {
      pass_count++;
    }
  }

  CHECK_EQ(pass_count, 512);
  std::cout << "  Tested 512 elements, all passed\n";
  std::cout << "[Pass] Deep 3D tensor (8x8x8) test\n\n";
}

void TestSigmoid_OutputRange() {
  std::cout << "--- Test 16: Sigmoid output range verification ---\n";

  auto input = std::make_shared<Tensor<float>>(1, 1, 10);
  auto output = std::make_shared<Tensor<float>>(1, 1, 10);

  // Test extreme values to verify output is always in (0, 1)
  std::vector<float> values = {-1000.0f, -100.0f, -10.0f, -1.0f, 0.0f, 1.0f, 10.0f, 100.0f, 1000.0f, 0.5f};
  input->Fill(values, false);

  auto activation_func = ApplySSEActivation(ActivationType::ActivationSigmoid);
  activation_func(input, output);

  for (int i = 0; i < 10; ++i) {
    float actual = output->index(i);
    // For extreme values, sigmoid approaches 0 or 1
    // Use tolerance for boundary checks
    if (values[i] < -100.0f) {
      CHECK(FloatEq(actual, 0.0f, 1e-3f));
    } else if (values[i] > 100.0f) {
      CHECK(FloatEq(actual, 1.0f, 1e-3f));
    } else {
      CHECK(actual >= 0.0f && actual <= 1.0f);
    }
    std::cout << "  Sigmoid(" << values[i] << ") = " << actual << " (in range [0,1])\n";
  }
  std::cout << "[Pass] Output range verification\n\n";
}

int main() {
  std::cout << "========================================\n";
  std::cout << "  Sigmoid Activation Function Tests\n";
  std::cout << "========================================\n\n";

  TestSigmoid_Nullptr();
  TestSigmoid_EmptyTensor();
  TestSigmoid_SingleElement();
  TestSigmoid_PositiveValues();
  TestSigmoid_NegativeValues();
  TestSigmoid_LargeValues();
  TestSigmoid_2DTensor();
  TestSigmoid_3DTensor();
  TestSigmoid_RandomValues();
  TestSigmoid_InPlace();
  TestSigmoid_MismatchedSize();
  TestSigmoid_BatchProcessing();
  TestSigmoid_SpecialValues();
  TestSigmoid_LargeTensor();
  TestSigmoid_VeryLargeTensor();
  TestSigmoid_Deep3DTensor();
  TestSigmoid_OutputRange();

  std::cout << "========================================\n";
  std::cout << "  All Sigmoid tests completed!\n";
  std::cout << "========================================\n";

  return 0;
}
