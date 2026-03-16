/**
 * @file bench.cpp
 * @brief Performance benchmarks for DL4Cpp Tensor library using ubench.h
 * @details This file contains benchmark tests for tensor operations including
 *          element-wise arithmetic, broadcasting, linear algebra, and memory
 *          access patterns. All benchmarks use the ubench micro-benchmarking
 *          framework.
 */

// UBENCH_IMPLEMENTATION must be defined before including ubench.h
#define UBENCH_IMPLEMENTATION
#include "utils/ubench.h"

#include "core/tensor.hpp"
#include "core/tensor_math.hpp"
#include "core/tensor_linalg.hpp"
#include <vector>
#include <cmath>

using namespace ctl;
using namespace ctl::math;
using namespace ctl::linalg;

// ===== Simple Tensor Operations Benchmarks =====

UBENCH(Tensor, Create) {
  // Benchmark tensor construction with 100x100x100 dimensions
  Tensor<float> tensor(100, 100, 100);
}

UBENCH_EX(Tensor, Fill) {
  Tensor<float> tensor(100, 100, 100);
  std::vector<float> values(1000000, 1.0f);

  UBENCH_DO_BENCHMARK() {
    tensor.Fill(values, true);
  }
}

UBENCH_EX(Tensor, Rand) {
  Tensor<float> tensor(100, 100, 100);

  UBENCH_DO_BENCHMARK() {
    tensor.Rand();
  }
}

UBENCH_EX(Tensor, Reshape) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    tensor.Reshape({10000, 100, 1}, true);
  }
}

UBENCH_EX(Tensor, Flatten) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    tensor.Flatten(true);
  }
}

// ===== Element-wise Operations Benchmarks =====

UBENCH_EX(Math, Add_ElementWise) {
  Tensor<float> a(100, 100, 100);
  Tensor<float> b(100, 100, 100);
  a.Rand();
  b.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = add(a, b);
  }
}

UBENCH_EX(Math, Sub_ElementWise) {
  Tensor<float> a(100, 100, 100);
  Tensor<float> b(100, 100, 100);
  a.Rand();
  b.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = sub(a, b);
  }
}

UBENCH_EX(Math, Mul_ElementWise) {
  Tensor<float> a(100, 100, 100);
  Tensor<float> b(100, 100, 100);
  a.Rand();
  b.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = mul(a, b);
  }
}

UBENCH_EX(Math, Div_ElementWise) {
  Tensor<float> a(100, 100, 100);
  Tensor<float> b(100, 100, 100);
  a.Rand();
  b.Fill(1.0f); // Avoid division by zero

  UBENCH_DO_BENCHMARK() {
    auto result = div(a, b);
  }
}

// ===== Broadcasting Operations Benchmarks =====

UBENCH_EX(Math, Add_Broadcast) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();
  Tensor<float> bias(100, 1, 1);
  bias.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = add(tensor, bias);
  }
}

UBENCH_EX(Math, Sub_Broadcast) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();
  Tensor<float> bias(100, 1, 1);
  bias.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = sub(tensor, bias);
  }
}

UBENCH_EX(Math, Mul_Broadcast) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();
  Tensor<float> scale(100, 1, 1);
  scale.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = mul(tensor, scale);
  }
}

UBENCH_EX(Math, Div_Broadcast) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();
  Tensor<float> divisor(100, 1, 1);
  divisor.Fill(2.0f);

  UBENCH_DO_BENCHMARK() {
    auto result = div(tensor, divisor);
  }
}

// ===== Scalar Operations Benchmarks =====

UBENCH_EX(Math, AddScalar) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = add(tensor, 1.0f);
  }
}

UBENCH_EX(Math, SubScalar) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = sub(tensor, 1.0f);
  }
}

UBENCH_EX(Math, MulScalar) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = mul(tensor, 2.0f);
  }
}

UBENCH_EX(Math, DivScalar) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = div(tensor, 2.0f);
  }
}

// ===== Advanced Element-wise Operations Benchmarks =====

UBENCH_EX(Math, Exp) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = exp(tensor);
  }
}

UBENCH_EX(Math, Clip) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = clip(tensor, 0.0f, 1.0f);
  }
}

UBENCH_EX(Math, Transform_ReLU) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    tensor.Transform([](float x) { return std::max(0.0f, x); });
  }
}

UBENCH_EX(Math, Transform_Sigmoid) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    tensor.Transform([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
  }
}

UBENCH_EX(Math, Matmul) {
  Tensor<float> a(100, 50, 40);
  Tensor<float> b(100, 40, 60);
  a.Rand();
  b.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = matmul(a, b);
  }
}

// ===== In-place Operations Benchmarks =====

UBENCH_EX(Inplace, Add) {
  Tensor<float> a(100, 100, 100);
  Tensor<float> b(100, 100, 100);
  Tensor<float> out(100, 100, 100);
  a.Rand();
  b.Rand();

  UBENCH_DO_BENCHMARK() {
    ElementAdd(a, b, out);
  }
}

UBENCH_EX(Inplace, Sub) {
  Tensor<float> a(100, 100, 100);
  Tensor<float> b(100, 100, 100);
  Tensor<float> out(100, 100, 100);
  a.Rand();
  b.Rand();

  UBENCH_DO_BENCHMARK() {
    ElementSub(a, b, out);
  }
}

UBENCH_EX(Inplace, Multiply) {
  Tensor<float> a(100, 100, 100);
  Tensor<float> b(100, 100, 100);
  Tensor<float> out(100, 100, 100);
  a.Rand();
  b.Rand();

  UBENCH_DO_BENCHMARK() {
    ElementMultiply(a, b, out);
  }
}

UBENCH_EX(Inplace, Divide) {
  Tensor<float> a(100, 100, 100);
  Tensor<float> b(100, 100, 100);
  Tensor<float> out(100, 100, 100);
  a.Rand();
  b.Fill(2.0f);

  UBENCH_DO_BENCHMARK() {
    ElementDivide(a, b, out);
  }
}

// ===== Linear Algebra Benchmarks =====

UBENCH_EX(LinAlg, Euclidean_Norm) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    float result = norm(tensor);
    UBENCH_DO_NOTHING(&result);
  }
}

UBENCH_EX(LinAlg, L1_Norm) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    float result = norm1(tensor);
    UBENCH_DO_NOTHING(&result);
  }
}

UBENCH_EX(LinAlg, Dot_Product) {
  Tensor<float> a(100, 100, 100);
  Tensor<float> b(100, 100, 100);
  a.Rand();
  b.Rand();

  UBENCH_DO_BENCHMARK() {
    float result = dot(a, b);
    UBENCH_DO_NOTHING(&result);
  }
}

UBENCH_EX(LinAlg, Determinant) {
  // 10x10 square matrices with 32 channels
  Tensor<float> tensor(32, 10, 10);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = det(tensor);
  }
}

UBENCH_EX(LinAlg, Trace) {
  // 50x50 square matrices with 64 channels
  Tensor<float> tensor(64, 50, 50);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = trace(tensor);
  }
}

UBENCH_EX(LinAlg, Transpose) {
  Tensor<float> tensor(64, 100, 80);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = transpose(tensor);
  }
}

UBENCH_EX(LinAlg, Inverse) {
  // 20x20 invertible square matrices with 16 channels
  Tensor<float> tensor(16, 20, 20);
  // Fill with identity-like matrix to ensure invertibility
  for (uint32_t c = 0; c < 16; ++c) {
    for (uint32_t i = 0; i < 20; ++i) {
      for (uint32_t j = 0; j < 20; ++j) {
        tensor.at(c, i, j) = (i == j) ? 1.0f : 0.01f * (rand() % 100);
      }
    }
  }

  UBENCH_DO_BENCHMARK() {
    auto result = inv(tensor);
  }
}

UBENCH_EX(LinAlg, Outer_Product) {
  Tensor<float> a(50);
  Tensor<float> b(80);
  a.Rand();
  b.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = outer(a, b);
  }
}

// ===== Memory Access Pattern Tests =====

UBENCH_EX(Access, RowMajor) {
  Tensor<float> tensor(100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    float sum = 0.0f;
    for (size_t i = 0; i < 100; i++) {
      for (size_t j = 0; j < 100; j++) {
        sum += tensor.posi(i, j);
      }
    }
    UBENCH_DO_NOTHING(&sum);
  }
}

UBENCH_EX(Access, ColMajor) {
  Tensor<float> tensor(100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    float sum = 0.0f;
    for (size_t j = 0; j < 100; j++) {
      for (size_t i = 0; i < 100; i++) {
        sum += tensor.posi(i, j);
      }
    }
    UBENCH_DO_NOTHING(&sum);
  }
}

UBENCH_EX(Access, ChannelAccess) {
  Tensor<float> tensor(32, 100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    float sum = 0.0f;
    for (uint32_t c = 0; c < 32; ++c) {
      const auto &slice = tensor.slice(c);
      sum += arma::accu(slice);
    }
    UBENCH_DO_NOTHING(&sum);
  }
}

// ===== Large Scale Benchmarks =====

UBENCH_EX(Large, Add_512x512x64) {
  Tensor<float> a(64, 512, 512);
  Tensor<float> b(64, 512, 512);
  a.Rand();
  b.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = add(a, b);
  }
}

UBENCH_EX(Large, Broadcast_512x512x64) {
  Tensor<float> tensor(64, 512, 512);
  tensor.Rand();
  Tensor<float> bias(64, 1, 1);
  bias.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = add(tensor, bias);
  }
}

UBENCH_EX(Large, Matmul_512x512x64) {
  Tensor<float> a(64, 512, 256);
  Tensor<float> b(64, 256, 512);
  a.Rand();
  b.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = matmul(a, b);
  }
}

UBENCH_EX(Large, Determinant_32x32x128) {
  Tensor<float> tensor(128, 32, 32);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = det(tensor);
  }
}

UBENCH_EX(Large, Transpose_128x512x256) {
  Tensor<float> tensor(128, 512, 256);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = transpose(tensor);
  }
}

// ===== Activation Function Benchmarks =====

UBENCH_EX(Activation, ReLU) {
  Tensor<float> tensor(64, 100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    tensor.Transform([](float x) { return std::max(0.0f, x); });
  }
}

UBENCH_EX(Activation, Sigmoid) {
  Tensor<float> tensor(64, 100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    tensor.Transform([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
  }
}

UBENCH_EX(Activation, Tanh) {
  Tensor<float> tensor(64, 100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    tensor.Transform([](float x) { return std::tanh(x); });
  }
}

UBENCH_EX(Activation, LeakyReLU) {
  Tensor<float> tensor(64, 100, 100);
  tensor.Rand();
  constexpr float alpha = 0.01f;

  UBENCH_DO_BENCHMARK() {
    tensor.Transform([alpha](float x) { return x > 0 ? x : alpha * x; });
  }
}

UBENCH_MAIN()
