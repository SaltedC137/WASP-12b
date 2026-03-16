/**
 * @author Aska Lyn
 * @brief Learn how to implement tensors in C++
 * @date 2026-03-04 11:10:27
 */

#include "core/tensor_math.hpp"
#include "utils/check.hpp"
#include <algorithm>
#include <armadillo>
#include <cstddef>
#include <cstdint>

namespace ctl::math {

// A couple of tensors operations

template <typename OpFunc_dt>
void ElementOp_o(const ften &tensor1, const ften &tensor2, ften &output,
                 OpFunc_dt op) {
  CHECK(!tensor1.empty() && !tensor2.empty() && !output.empty());

  if (tensor1.shapes() == tensor2.shapes()) {
    CHECK(tensor1.shapes() == output.shapes()) << "Output shape mismatch";

    const float *ptr_1 = tensor1.data().memptr();
    const float *ptr_2 = tensor2.data().memptr();
    float *ptr_out = output.data().memptr();
    const size_t total_size = tensor1.data().n_elem;

#pragma omp parallel for
    for (size_t i = 0; i < total_size; ++i) {
      ptr_out[i] = op(ptr_1[i], ptr_2[i]);
    }
    return;
  }

  // Broadcasting: tensor1 (C,H,W) + tensor2 (C,1,1) -> output (C,H,W)
  CHECK_EQ(tensor1.channels(), tensor2.channels()) << "channels mustmatch !";
  CHECK_EQ(tensor2.rows(), 1) << " only supports 1D biascurrently.";
  CHECK_EQ(tensor2.cols(), 1) << " only supports 1D biascurrently.";
  CHECK(tensor1.shapes() == output.shapes())
      << "Output shape must match tensor1.";

  const uint32_t channels = tensor1.channels();
  const uint32_t spatio_size = tensor1.rows() * tensor1.cols();

  const float *ptr_1 = tensor1.data().memptr();
  const float *ptr_bias = tensor2.data().memptr();
  float *ptr_out = output.data().memptr();

#pragma omp parallel for collapse(2)
  for (uint32_t c = 0; c < channels; ++c) {
    for (uint32_t s = 0; s < spatio_size; ++s) {
      uint32_t idx = c * spatio_size + s;
      ptr_out[idx] = op(ptr_1[idx], ptr_bias[c]);
    }
  }
}

void ElementAdd(const ften &tensor1, const ften &tensor2, ften &output) {
  ElementOp_o(tensor1, tensor2, output, [](float a, float b) { return a + b; });
}

void ElementSub(const ften &tensor1, const ften &tensor2, ften &output) {
  ElementOp_o(tensor1, tensor2, output, [](float a, float b) { return a - b; });
}

void ElementMultiply(const ften &tensor1, const ften &tensor2, ften &output) {
  ElementOp_o(tensor1, tensor2, output, [](float a, float b) { return a * b; });
}

void ElementDivide(const ften &tensor1, const ften &tensor2, ften &output) {
  ElementOp_o(tensor1, tensor2, output, [](float a, float b) {
    CHECK(std::abs(b) > 1e-6) << "Division by zero encountered!";
    return a / b;
  });
}

void Matmul(const ften &tensor1, const ften &tensor2, ften &output) {
  CHECK(!tensor1.empty() && !tensor2.empty() && !output.empty());

  CHECK_EQ(tensor1.cols(), tensor2.rows())
      << "Matmul dimension mismatch: K dimension must align!";
  CHECK_EQ(output.rows(), tensor1.rows()) << "Output M dimension mismatch!";
  CHECK_EQ(output.cols(), tensor2.cols()) << "Output N dimension mismatch!";

  const uint32_t channels = tensor1.channels();
  CHECK_EQ(channels, tensor2.channels())
      << "Batched Matmul requires matching channels!";
  CHECK_EQ(channels, output.channels())
      << "Output channels must match input channels!";

#pragma omp parallel for if (channels > 1)
  for (uint32_t c = 0; c < channels; ++c) {
    output.slice(c).zeros();
    output.slice(c) += tensor1.slice(c) * tensor2.slice(c);
  }
}

// Scalar operations

template <typename OpFunc_sc>
void ScalarOp_s(const ften &tensor, float scalar, ften &output, OpFunc_sc op) {
  CHECK(!tensor.empty() && !output.empty());
  CHECK(tensor.shapes() == output.shapes()) << "Shape mismatch in ScalarOp";

  const float *in_ptr = tensor.data().memptr();
  float *out_ptr = output.data().memptr();
  const size_t total_size = tensor.data().n_elem;

#pragma omp parallel for if (total_size > 8192)
  for (uint32_t i = 0; i < total_size; ++i) {
    out_ptr[i] = op(in_ptr[i], scalar);
  }
}
void AddScalar(const ften &tensor, float scalar, ften &output) {
  ScalarOp_s(tensor, scalar, output, [](float a, float b) { return a + b; });
}

void SubScalar(const ften &tensor, float scalar, ften &output) {
  ScalarOp_s(tensor, scalar, output, [](float a, float b) { return a - b; });
}

void MultiplyScalar(const ften &tensor, float scalar, ften &output) {
  ScalarOp_s(tensor, scalar, output, [](float a, float b) { return a * b; });
}

void DivideScalar(const ften &tensor, float scalar, ften &output) {
  ScalarOp_s(tensor, scalar, output, [](float a, float b) {
    CHECK(std::abs(b) > 1e-6) << "Division by zero encountered!";
    return a / b;
  });
}

void ElementExp(const ften &tensor, ften &output) {
  CHECK(!tensor.empty() && !output.empty());
  CHECK(tensor.shapes() == output.shapes()) << "Shape mismatch in ElementExp";

  output.data() = arma::exp(tensor.data());
}

void ElementClip(const ften &tensor, float min_val, float max_val,
                 ften &output) {
  CHECK(!tensor.empty() && !output.empty());
  CHECK(tensor.shapes() == output.shapes()) << "Shape mismatch in ElementClip";

  CHECK(min_val < max_val);

  const float *in_ptr = tensor.data().memptr();
  float *out_ptr = output.data().memptr();
  const size_t total_size = tensor.data().n_elem;

#pragma omp parallel for if (total_size > 8192)
  for (size_t i = 0; i < total_size; ++i) {
    out_ptr[i] = std::clamp(in_ptr[i], min_val, max_val);
  }
}
} // namespace ctl::math
