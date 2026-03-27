

#include "softmax.hpp"
#include "check.hpp"
#include "fmath.hpp"
#include "log.hpp"
#include "rt_type.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

namespace ctl {
namespace nn {

SoftmaxLayer::SoftmaxLayer(int32_t dim)
    : NoneParamLayer("Softmax"), softmax_dim_(dim) {}

StatusCode
SoftmaxLayer::Forward(const std::vector<std::shared_ptr<ften>> &inputs,
                      std::vector<std::shared_ptr<ften>> outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "SoftmaxLayer::Forward: inputs is empty";
    return StatusCode::InferInputsEmpty;
  }
  if (outputs.empty()) {
    LOG(ERROR) << "SoftmaxLayer::Forward: outputs is empty";
    return StatusCode::InferOutputsEmpty;
  }
  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "SoftmaxLayer::Forward: inputs and outputs size mismatch";
    return StatusCode::InferDimMismatch;
  }
  const uint32_t batch_size = inputs.size();
#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; i++) {
    const std::shared_ptr<ften> &input = inputs.at(i);
    CHECK(input != nullptr && input->empty())
        << "The input tensor array in the softmax layer has an empty tensor "
        << i << " th";

    std::shared_ptr<ften> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(i) = output;
    }
    CHECK(input->shapes() == output->shapes())
        << "The input and output tensor shapes of the softmax layer do not "
           "match "
        << i << " th";
    int32_t dim = this->softmax_dim_;
    std::vector<uint32_t> raw_shapes = input->shapes();

    if (dim < 0) {
      dim += int32_t(raw_shapes.size());
    }

    if (dim < 0 || dim >= 3 || dim > raw_shapes.size()) {
      LOG(FATAL) << "Error softmax dimension, which need between 0 and 2, "
                    "but dimension is "
                 << dim;
    }

    if (raw_shapes.size() == 1 && dim == 0) {
      float *input_ptr = input->raw_ptr();
      float *output_ptr = output->raw_ptr();

      int32_t size = static_cast<int32_t>(raw_shapes.front());
      float max_val = *std::max_element(input_ptr, input_ptr + size);

      int32_t index = 0;
      float sum_value = 0.f;

#ifdef __AVX2__
      int32_t pack_size = 8;
      input_ptr = input->raw_ptr();
      __m256 sum_vec = _mm256_setzero_ps();
      __m256 max_value256 = _mm256_set1_ps(max_val);
      for (; index <= size - pack_size; index += pack_size) {
        __m256 p = _mm256_loadu_ps(input_ptr);
        __m256 exp_sub_value = fmath::exp_ps256(_mm256_sub_ps(p, max_value256));
        _mm256_storeu_ps(output_ptr, exp_sub_value);
        sum_vec = _mm256_add_ps(sum_vec, exp_sub_value);
        input_ptr += pack_size;
        output_ptr += pack_size;
      }
      sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
      sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
      sum_value = ((float *)&sum_vec)[0] + ((float *)&sum_vec)[4];
#endif

      if (index < size) {
        input_ptr = input->raw_ptr(index);
        output_ptr = output->raw_ptr(index);
        for (; index < size; ++index) {
          float cur_value = *input_ptr;
          float exp_sub_value = fmath::exp(cur_value - max_val);
          *output_ptr = exp_sub_value;
          sum_value += exp_sub_value;

          input_ptr += 1;
          output_ptr += 1;
        }
      }

      index = 0;

#ifdef __AVX2__
      output_ptr = output->raw_ptr();
      sum_vec = _mm256_set1_ps(sum_value);
      for (; index <= size - pack_size; index += pack_size) {
        __m256 p = _mm256_loadu_ps(output_ptr);
        __m256 div_value = _mm256_div_ps(p, sum_vec);
        _mm256_storeu_ps(output_ptr, div_value);
        output_ptr += pack_size;
      }
#endif
      if (index < size) {
        output_ptr = output->raw_ptr(index);
        for (; index < size; ++index) {
          *output_ptr = *output_ptr / sum_value;
          output_ptr += 1;
        }
      }
    } else {
      const uint32_t padding_size_num = 3 - raw_shapes.size();
      for (uint32_t j = 0; j < padding_size_num; j++) {
        raw_shapes.push_back(1);
      }

      // trans dim
      // input
      const uint32_t inner_sizes = std::accumulate(
          raw_shapes.begin() + dim + 1, raw_shapes.end(), 1, std::multiplies());

      // output

      const uint32_t outer_sizes = std::accumulate(
          raw_shapes.begin(), raw_shapes.begin() + dim, 1, std::multiplies());

      int32_t axis_sizes = static_cast<int32_t>(raw_shapes.at(dim));
      CHECK_EQ(axis_sizes * outer_sizes * inner_sizes, input->size());
      std::vector<float> output_values = input->values(true);

#pragma omp parallel for collapse(2)
      for (int32_t outer_size = 0; outer_size < outer_sizes; outer_size++) {
        for (int32_t inner_size = 0; inner_size < inner_sizes; inner_size++) {
          float max_val = std::numeric_limits<float>::lowest();


          
        }
      }
    }
  }
  return StatusCode::Success;
}

} // namespace nn
} // namespace ctl