/**
 * @file bench.cpp
 * @brief Example benchmarks using ubench.h
 */

// UBENCH_IMPLEMENTATION must be defined before including ubench.h
#define UBENCH_IMPLEMENTATION
#include "ubench.h"

#include "tensor.hpp"
#include "tensor_math.hpp"
#include <vector>

using namespace ctl;
using namespace ctl::math;

// ===== Simple Benchmarks =====

// Method 1: UBENCH(SET, NAME) - Simplest form
UBENCH(Tensor, Create) {
  Tensor<float> tensor(100, 100, 100);
}

// Method 2: UBENCH_EX(SET, NAME) - More flexible, access to ubench_run_state
UBENCH_EX(Tensor, Fill) {
  Tensor<float> tensor(100, 100, 100);
  std::vector<float> values(1000000, 1.0f);

  UBENCH_DO_BENCHMARK() {
    tensor.Fill(values, true);
  }
}

// ===== Fixture-based Benchmarks =====

// Define a fixture struct
struct TensorFixture {
  Tensor<float> tensor;
  std::vector<float> data;

  void Setup() {
    tensor = Tensor<float>(50, 50, 50);
    data = std::vector<float>(125000, 1.0f);
  }

  void Teardown() {
    // Cleanup code (if needed)
  }
};

// Fixture setup function
UBENCH_F_SETUP(TensorFixture) {
  ubench_fixture->Setup();
}

// Fixture teardown function
UBENCH_F_TEARDOWN(TensorFixture) {
  ubench_fixture->Teardown();
}

// Benchmark using fixture
UBENCH_F(TensorFixture, Add) {
  Tensor<float> other(50, 50, 50);
  other.Fill(2.0f);

  auto result = add(ubench_fixture->tensor, other);
}

// ===== Math Operations Tests =====

UBENCH(Math, Add3D) {
  Tensor<float> a(100, 100, 10);
  Tensor<float> b(100, 100, 10);
  a.Rand();
  b.Rand();

  auto result = add(a, b);
}

UBENCH(Math, AddScalarBroadcast) {
  // Test broadcasting: 3D tensor + per-channel bias
  Tensor<float> tensor(100, 50, 50);
  tensor.Rand();
  Tensor<float> bias(100, 1, 1);
  bias.Rand();

  auto result = add(tensor, bias);
}

UBENCH(Math, Transform) {
  Tensor<float> tensor(100, 100, 10);
  tensor.Rand();

  tensor.Transform([](float x) {
    return x > 0.5f ? 1.0f : 0.0f;
  });
}

// ===== Memory Access Pattern Tests =====

UBENCH_EX(Access, RowMajor) {
  Tensor<float> tensor(100, 100);
  tensor.Rand();

  float sum = 0.0f;
  UBENCH_DO_BENCHMARK() {
    // Row-major access
    for (size_t i = 0; i < 100; i++) {
      for (size_t j = 0; j < 100; j++) {
        sum += tensor.at(0, i, j);
      }
    }
    // Prevent compiler optimization
    UBENCH_DO_NOTHING(&sum);
  }
}

UBENCH_EX(Access, ColMajor) {
  Tensor<float> tensor(100, 100);
  tensor.Rand();

  float sum = 0.0f;
  UBENCH_DO_BENCHMARK() {
    // Column-major access (may be slower)
    for (size_t j = 0; j < 100; j++) {
      for (size_t i = 0; i < 100; i++) {
        sum += tensor.at(0, i, j);
      }
    }
    UBENCH_DO_NOTHING(&sum);
  }
}

// Define global state required by ubench and start main function
UBENCH_MAIN()
