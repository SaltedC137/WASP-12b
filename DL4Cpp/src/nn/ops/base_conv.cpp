

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


  

  return StatusCode::Success;
}

} // namespace nn
} // namespace ctl