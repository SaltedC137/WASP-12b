

#include "nn/ops/sigmoid.hpp"
#include "check.hpp"
#include "nn/ops/activation.hpp"
#include <cstdint>

namespace ctl {
namespace nn {

static void SigmoidSSE(sften input, sften output) {
  CHECK(input != nullptr && output != nullptr) << "Tensor is null";
  CHECK(!input->empty() && !output->empty()) << "Tensor is empty";
  CHECK(input->size() == output->size()) << "Tensor size not match";
  int64_t index = 0;
  int64_t packet_size;
  int64_t in_size = static_cast<int64_t>(input->size());
  const float *in_ptr = input->raw_ptr();
  float *out_ptr = output->raw_ptr();
#ifdef __AVX2__
  packet_size = 8;
  __m256 one = _mm256_set1_ps(1.f);
  





#endif
}
} // namespace nn
} // namespace ctl
