

#include "nn/ops/sigmoid.hpp"
#include "check.hpp"
#include "log.hpp"
#include "nn/ops/activation.hpp"
#include "utils/fmath.hpp"
#include <cstdint>
#include <immintrin.h>

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

// magic
#ifdef __AVX2__
  int64_t limit_avx = in_size - (in_size % 8);
#pragma omp parallel
  {
    __m256 one = _mm256_set1_ps(1.f);
    __m256 zero = _mm256_setzero_ps();
#pragma omp for
    for (int64_t j = 0; j < limit_avx; j += 8) {
      __m256 p = _mm256_loadu_ps(in_ptr + j);
      p = _mm256_div_ps(
          one, _mm256_add_ps(one, fmath::exp_ps256(_mm256_sub_ps(zero, p))));
      _mm256_storeu_ps(out_ptr + j, p);
    }
  }
  index = limit_avx;
#endif

#ifdef __SSE2__

  int64_t limit_sse = in_size - ((in_size - index) % 4);
  __m128 one128 = _mm_set1_ps(1.f);
  __m128 zero128 = _mm_setzero_ps();

  for (; index < limit_sse; index += 4) {
    __m128 p = _mm_loadu_ps(in_ptr + index);
    p = _mm_div_ps(one128,
                   _mm_add_ps(one128, fmath::exp_ps(_mm_sub_ps(zero128, p))));
    _mm_storeu_ps(out_ptr + index, p);
  }

#endif

  for (; index < in_size; ++index) {
    float value = input->index(index);
    output->index(index) = 1.f / (1.f + fmath::exp(-value));
  }
}


void Sigmoid::operator()(const sften& input, const sften& output) const {
  SigmoidSSE(input, output);
}


ActivationFunc ApplySSEActivation(ActivationType act_type) {
  ActivationFunc function;
  switch (act_type) {
  case ActivationType::ActivationSigmoid: {
    function = SigmoidSSE;
    return function;
  }
  default: {
    LOG(FATAL) << "Unknown SSE activation type: " << int32_t(act_type);
  }
  }
  return function;
}

} // namespace nn
} // namespace ctl