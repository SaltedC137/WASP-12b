/**
 * @author Aska Lyn
 * @brief Learn how to implement tensors in C++
 * @date 2026-03-04 11:10:27
 */

#include "tensor_math.hpp"
#include "check.hpp"
#include <cstdint>

namespace ctl::math {

void ElementAdd(const ften &tensor1, const ften &tensor2, ften &output) {
  CHECK(!tensor1.empty() && !tensor2.empty() && !output.empty());

  if (tensor1.shapes() == tensor2.shapes()) {
    CHECK(tensor1.shapes() == output.shapes()) << "Output shape mismatch";
    output.data() = tensor1.data() + tensor2.data();
    return;
  }

  // Broadcasting: tensor1 (C,H,W) + tensor2 (C,1,1) -> output (C,H,W)
  CHECK_EQ(tensor1.channels(), tensor2.channels()) << "channels must match!";
  CHECK_EQ(tensor2.rows(), 1) << "only supports 1D bias currently.";
  CHECK_EQ(tensor2.cols(), 1) << "only supports 1D bias currently.";
  CHECK(tensor1.shapes() == output.shapes())
      << "Output shape must match tensor1.";

  const uint32_t channels = tensor1.channels();
  const uint32_t rows = tensor1.rows();
  const uint32_t cols = tensor1.cols();

#pragma omp parallel for if (channels > 3)
  for (uint32_t c = 0; c < channels; c++) {
    // Get bias value for this channel
    float bias_val = tensor2.slice(c)(0, 0);
    // Add scalar to entire slice
    output.slice(c) = tensor1.slice(c) + bias_val;
  }
}

void ElementSub(const ften &tensor1, const ften &tensor2, ften &output) {
  CHECK(!tensor1.empty() && !tensor2.empty() && !output.empty());

  if (tensor1.shapes() == tensor2.shapes()) {
    CHECK(tensor1.shapes() == output.shapes()) << "Output shape mismatch";
    output.data() = tensor1.data() - tensor2.data();
    return;
  }

  CHECK_EQ(tensor1.channels(), tensor2.channels()) << "channels must march";
  CHECK_EQ(tensor2.rows(), 1) << "only supports 1D bias currently.";
  CHECK_EQ(tensor2.cols(), 1) << "only supports 1D bias currently.";

  CHECK(tensor1.shapes() == output.shapes())
      << "Output shape must match tensor1.";

  const uint32_t channels = tensor1.channels();
  const uint32_t rows = tensor1.rows();
  const uint32_t cols = tensor1.cols();

#pragma omp parallel for if (channels > 3)

  for (uint32_t c = 0; c < channels; c++) {
    float bias_val = tensor2.slice(c)(0, 0);
    output.slice(c) = tensor1.slice(c) - bias_val;
  }
}


void ElementDivide(const ften &tensor1, const ften &tensor2, ften &output){
  CHECK(!tensor1.empty() && !tensor2.empty() && !output.empty());


}

void ElementMultiply(const ften &tensor1, const ften &tensor2, ften &output){}



} // namespace ctl::math