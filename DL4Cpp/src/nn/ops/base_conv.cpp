

#include "nn/ops/base_conv.hpp"
#include "check.hpp"
#include "log.hpp"
#include "param_layer.hpp"
#include "rt_type.hpp"
#include "tensor.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace ctl {
namespace nn {

BaseConvolutionLayer::BaseConvolutionLayer(
    ConvType conv_type, uint32_t output_channel, uint32_t in_channel,
    uint32_t kernel_h, uint32_t kernel_w, uint32_t padding_h,
    uint32_t padding_w, uint32_t stride_h, uint32_t stride_w, uint32_t groups,
    bool use_bias, uint32_t output_padding_h, uint32_t output_padding_w,
    uint32_t dilation_h, uint32_t dilation_w)
    : ParamLayer("convolution"), conv_type_(conv_type), use_bias_(use_bias),
      groups_(groups), padding_h_(padding_h), padding_w_(padding_w),
      stride_h_(stride_h), stride_w_(stride_w),
      output_padding_h_(output_padding_h), output_padding_w_(output_padding_w),
      dilation_h_(dilation_h), dilation_w_(dilation_w) {
  if (groups != 1) {
    in_channel /= groups;
  }
  CHECK_GE(groups_, 1);
  CHECK_GT(kernel_h, 0);
  CHECK_GT(kernel_w, 0);
  CHECK_GT(stride_h_, 0);
  CHECK_GT(stride_w_, 0);
  CHECK_GT(dilation_h, 0);
  CHECK_GT(dilation_w, 0);

  if (conv_type_ == ConvType::OpConv) {
    CHECK_EQ(output_padding_h_, 0);
    CHECK_EQ(output_padding_w_, 0);
  } else if (conv_type_ == ConvType::OpDeconv) {
    if (dilation_h > 1) {
      kernel_h = kernel_h = (kernel_h - 1) * (dilation_h_ - 1) + kernel_h;
    }
    if (dilation_w > 1) {
      kernel_w = (kernel_w - 1) * (dilation_w_ - 1) + kernel_w;
    } else {
      LOG(FATAL) << "dilation_w should be greater than | 1 value:"
                 << int(conv_type_);
    }
    CHECK_GT(kernel_h, 0);
    CHECK_GT(kernel_w, 0);
    this->InitWeightParam(output_channel, in_channel, kernel_h, kernel_w);
    if (use_bias_) {
      this->InitBiasParam(output_channel, 1, 1, 1);
    }
  }
}

void BaseConvolutionLayer::InitIm2ColWeight() {}

void BaseConvolutionLayer::AddBias(arma::fmat &output,
                                   uint32_t bias_index) const {
  if (!this->bias_.empty() && this->use_bias_) {
    std::shared_ptr<ften> bias;
    bias = this->bias_.at(bias_index);
    if (bias != nullptr && !bias->empty()) {
      float bias_value = bias->index(0);
      output += bias_value;
    } else {
      LOG(FATAL) << "Bias is empty or not initialized";
    }
  }
}

StatusCode
BaseConvolutionLayer::Forward(const std::vector<std::shared_ptr<ften>> &inputs,
                              std::vector<std::shared_ptr<ften>> &outputs) {

  StatusCode check_code = Check(inputs, outputs);
  if (check_code != StatusCode::Success) {
    return check_code;
  }

  const uint32_t kernel_count = this->weights_.size();
  const uint32_t kernel_h = this->weights_.at(0)->rows();
  const uint32_t kernel_w = this->weights_.at(0)->cols();
  const uint32_t kernel_channel = this->weights_.at(0)->channels();

  if (kernel_matrix_arr_.size() != kernel_count) {
    InitIm2ColWeight();
  }
  const uint32_t batch_size = inputs.size();
  const uint32_t kernel_count_group = kernel_count / groups_;

#pragma omp parallel for num_thread(batch_size)
  for (uint32_t i = 0; i < batch_size; i++) {
    const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
    const uint32_t input_h = input->rows();
    const uint32_t input_w = input->cols();
    const uint32_t input_c = input->channels();

    CHECK(input_h > 0 && input_w > 0 && input_c > 0);

    const auto &output_size =
        ComputeOutputSize(input_h, input_w, input_h, input_c);

    const uint32_t output_h = output_size.first;
    const uint32_t output_w = output_size.second;
    CHECK(output_h > 0 && output_w > 0)
        << " output_h or output_w is zero" << i << "th";

    std::shared_ptr<Tensor<float>> output_tensor = outputs.at(i);
    if (output_tensor == nullptr || output_tensor->empty()) {
      output_tensor =
          std::make_shared<Tensor<float>>(kernel_count, output_h, output_w);
      outputs.at(i) = output_tensor;
    }

    CHECK(output_tensor->rows() == output_h &&
          output_tensor->cols() == output_w &&
          output_tensor->channels() == kernel_count)
        << "output tensor size is not correct in:" << i << "th";

#pragma omp parallel for if (groups_ > 1)
    for (uint32_t group = 0; group < groups_; ++group) {
      if (groups_ != 1) {
        CHECK(kernel_count % groups_ == 0);
        CHECK(input_c % groups_ == 0);
      }
      const uint32_t channels_per_group = input_c / groups_;
      CHECK(channels_per_group == kernel_channel)
          << "The number of channel for the kernel "
             "matrix and input tensor do not match";
      ComputeOutput(input, output_tensor, kernel_h, kernel_w,
                    kernel_count_group, input_h, input_w, channels_per_group,
                    output_h, output_w, group);
    }
  }
  return StatusCode::Success;
}

} // namespace nn
} // namespace ctl