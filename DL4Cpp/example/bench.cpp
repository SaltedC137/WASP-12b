/**
 * @file bench.cpp
 * @brief Performance benchmarks for DL4Cpp Tensor library using ubench.h
 * @details This file contains benchmark tests for tensor operations including
 *          element-wise arithmetic, broadcasting, and memory access patterns.
 */

// UBENCH_IMPLEMENTATION must be defined before including ubench.h
#define UBENCH_IMPLEMENTATION
#include "ubench.h"

#include "tensor.hpp"
#include "tensor_math.hpp"
#include <vector>
#include <cmath>

using namespace ctl;
using namespace ctl::math;

// ===== Simple Benchmarks =====

UBENCH(Tensor, Create) {
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

UBENCH_EX(Math, Add_ScalarBroadcast) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();
  Tensor<float> bias(100, 1, 1);
  bias.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = add(tensor, bias);
  }
}

UBENCH_EX(Math, Sub_ScalarBroadcast) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();
  Tensor<float> bias(100, 1, 1);
  bias.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = sub(tensor, bias);
  }
}

UBENCH_EX(Math, Mul_ScalarBroadcast) {
  Tensor<float> tensor(100, 100, 100);
  tensor.Rand();
  Tensor<float> scale(100, 1, 1);
  scale.Rand();

  UBENCH_DO_BENCHMARK() {
    auto result = mul(tensor, scale);
  }
}

UBENCH_EX(Math, Div_ScalarBroadcast) {
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

// ===== Advanced Operations Benchmarks =====

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

// ===== Memory Access Pattern Tests =====

UBENCH_EX(Access, RowMajor) {
  Tensor<float> tensor(100, 100);
  tensor.Rand();

  UBENCH_DO_BENCHMARK() {
    float sum = 0.0f;
    for (size_t i = 0; i < 100; i++) {
      for (size_t j = 0; j < 100; j++) {
        sum += tensor.at(0, i, j);
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
        sum += tensor.at(0, i, j);
      }
    }
    UBENCH_DO_NOTHING(&sum);
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

UBENCH_MAIN()
