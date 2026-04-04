

#include "relu.hpp"
#include "activation.hpp"
#include "check.hpp"
#include "layer_factory.hpp"
#include "rt_type.hpp"
#include "tensor.hpp"
#include <cstdint>
#include <immintrin.h>
#include <memory>
#include <vector>

namespace ctl {
namespace nn {

static void ReLUAVX2(sften input, sften output) {
  CHECK(input != nullptr && output != nullptr)
      << "The input or output tensor is empty.";
  CHECK(!input->empty() && !output->empty())
      << "The input or output tensor is empty.";
  CHECK(input->size() == output->size())
      << "The input and output sizes are not equal.";

  const int64_t size = static_cast<int64_t>(input->size());
  const float *in_ptr = input->raw_ptr();
  float *out_ptr = output->raw_ptr();

  int64_t i = 0;

#ifdef __AVX2__

  const int64_t step = 16;
  const int64_t vec_size = size - (size % step);

#pragma omp parallel for schedule(static)
  for (int64_t j = 0; j < vec_size; j += step) {
    __m256 zero = _mm256_setzero_ps();

    __m256 p1 = _mm256_load_ps(in_ptr + j);
    __m256 p2 = _mm256_load_ps(in_ptr + j + 8);

    __m256 v1 = _mm256_max_ps(zero, p1);
    __m256 v2 = _mm256_max_ps(zero, p2);

    _mm256_storeu_ps(out_ptr + j, v1);
    _mm256_storeu_ps(out_ptr + j + 8, v2);
  }
  i = vec_size;

  for (; i <= size - 8; i += 8) {
    __m256 zero = _mm256_setzero_ps();
    __m256 p = _mm256_load_ps(in_ptr + i);
    _mm256_storeu_ps(out_ptr + i, _mm256_max_ps(zero, p));
  }
#elif __SSE2__

  const int64_t step = 16;
  const int64_t vec_size = size - (size % step);

#pragma omp parallel for schedule(static)
  for (int64_t j = 0; j < vec_size; j += step) {
    __m128 zero = _mm_setzero_ps();

    __m128 p1 = _mm_loadu_ps(in_ptr + j);
    __m128 p2 = _mm_loadu_ps(in_ptr + j + 4);
    __m128 p3 = _mm_loadu_ps(in_ptr + j + 8);
    __m128 p4 = _mm_loadu_ps(in_ptr + j + 12);

    _mm_storeu_ps(out_ptr + j, _mm_max_ps(zero, p1));
    _mm_storeu_ps(out_ptr + j + 4, _mm_max_ps(zero, p2));
    _mm_storeu_ps(out_ptr + j + 8, _mm_max_ps(zero, p3));
    _mm_storeu_ps(out_ptr + j + 12, _mm_max_ps(zero, p4));
  }
  i = vec_size;

  for (; i <= size - 4; i += 4) {
    __m128 zero = _mm_setzero_ps();
    _mm_storeu_ps(out_ptr + i, _mm_max_ps(zero, _mm_loadu_ps(in_ptr + i)));
  }

#endif
  for (; i < size; ++i) {
    out_ptr[i] = std::max(in_ptr[i], 0.f);
  }
}

void ReLU::operator()(const sften &input, sften &output) const {
  ReLUAVX2(input, output);
}

ReLULayer::ReLULayer()
    : ActivationLayer(ActivationType::ActivationRelu, "nn.ReLU") {}

StatusCode
ReLULayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                   std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  return ActivationLayer::Forward(inputs, outputs);
}

StatusCode
ReLULayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op,
                          std::shared_ptr<Layer<float>> &relu_layer) {
  if (!op) {
    LOG(ERROR) << "The relu operator parameter in the layer is null pointer.";
    return StatusCode::ParseNullOperator;
  }

  relu_layer = std::make_shared<ReLULayer>();
  return StatusCode::Success;
}


LayerRegisterWrapper ReLUCreateInstance(ReLULayer::CreateInstance, "nn.ReLU");


} // namespace nn
} // namespace ctl