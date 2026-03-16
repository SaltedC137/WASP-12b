/**
 * @author Aska Lyn
 * @brief Learn how to implement tensors in C++
 * @date 2026-03-14 12:39:20
 */

#include "utils/check.hpp"
#include "core/tensor.hpp"
#include <armadillo>
#include <cstdint>

namespace ctl::linalg {

float Euclidean_norm(const ften &tensor) {
  CHECK(!tensor.empty()) << "Cannot compute norm of an empty tensor!";

  arma::fvec vec_tes(const_cast<float *>(tensor.data().memptr()), tensor.size(),
                     false);
  return arma::norm(vec_tes, 2);
}

float Absolute_value_norm(const ften &tensor) {
  CHECK(!tensor.empty()) << "Cannot compute norm of an empty tensor!";

  arma::fvec vec_tes(const_cast<float *>(tensor.data().memptr()), tensor.size(),
                     false);
  return arma::norm(vec_tes, 1);
}

float Transvection(const ften &tensor1, const ften &tensor2) {
  CHECK(!tensor1.empty() && !tensor2.empty()) << "";
  CHECK_EQ(tensor1.size(), tensor2.size())
      << "Dot product requires equal total size!";

  arma::fvec vec1(const_cast<float *>(tensor1.data().memptr()), tensor1.size(),
                  false);
  arma::fvec vec2(const_cast<float *>(tensor2.data().memptr()), tensor2.size(),
                  false);

  return arma::dot(vec1, vec2);
}

void Batch_Determinant(const ften &tensor, ften &output) {
  CHECK(!tensor.empty());
  CHECK_EQ(tensor.rows(), tensor.cols())
      << "Determinant requires square matrices!";

  const uint32_t channels = tensor.channels();
  CHECK_EQ(output.size(), channels)
      << "Output tensor must have the same number of channels as the input "
         "tensor!";

  float *out_ptr = output.data().memptr();

#pragma omp parallel for if (channels > 1)
  for (uint32_t c = 0; c < channels; c++) {
    out_ptr[c] = arma::det(tensor.data().slice(c));
  }
}

void Batch_Trace(const ften &tensor, ften &output) {
  CHECK(!tensor.empty() && !output.empty());
  CHECK_EQ(tensor.rows(), tensor.cols());

  const uint32_t channels = tensor.channels();
  CHECK_EQ(output.size(), channels);

  float *out_ptr = output.data().memptr();

#pragma omp parallel for if (channels > 1)
  for (uint32_t c = 0; c < channels; c++) {
    out_ptr[c] = arma::trace(tensor.data().slice(c));
  }
}

void Batch_Inverse(const ften &tensor, ften &output) {
  CHECK(!tensor.empty() && !output.empty());
  CHECK_EQ(tensor.rows(), tensor.cols()) << "Inverse requires square matrices!";

  const uint32_t channels = tensor.channels();

#pragma omp parallel for if (channels > 1)
  for (uint32_t c = 0; c < channels; ++c) {
    output.data().slice(c) = arma::inv(tensor.data().slice(c));
  }
}

void Transposition(const ften &tensor, ften &output) {
  CHECK(!tensor.empty() && !output.empty());
  CHECK_EQ(tensor.channels(), output.channels());

  const uint32_t channels = tensor.channels();

#pragma omp parallel for if (channels > 1)
  for (uint32_t c = 0; c < channels; ++c) {
    output.data().slice(c) = tensor.data().slice(c).t();
  }
}

void Outer_product(const ften &tensor1, const ften &tensor2, ften &output) {
  CHECK(!tensor1.empty() && !tensor2.empty() && !output.empty());

  CHECK_EQ(output.rows(), tensor1.size()) << "Output rows must match tensor1 size!";
  CHECK_EQ(output.cols(), tensor2.size()) << "Output cols must match tensor2 size!";
  CHECK_EQ(output.channels(), 1)
      << "Outer product outputs a 2D matrix (1 channel)!";

  arma::fvec v1(const_cast<float *>(tensor1.data().memptr()), tensor1.size(),
                false);
  arma::fvec v2(const_cast<float *>(tensor2.data().memptr()), tensor2.size(),
                false);

  output.data().slice(0) = v1 * v2.t();
}
} // namespace ctl::linalg