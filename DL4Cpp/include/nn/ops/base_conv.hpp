

#include "param_layer.hpp"
#include "rt_type.hpp"
namespace ctl {
namespace nn {

enum class ConvType {
  OpConv = 0,
  OpConvKnow = -1,
  kOpDeconv = 1,
};

class BaseConvolutionLayer : public ParamLayer {
public:
  explicit BaseConvolutionLayer(
      ConvType conv_type, uint32_t output_channel, uint32_t in_channel,
      uint32_t kernel_h, uint32_t kernel_w, uint32_t padding_h,
      uint32_t padding_w, uint32_t stride_h, uint32_t stride_w, uint32_t groups,
      bool use_bias = true, uint32_t output_padding_h = 0,
      uint32_t output_padding_w = 0, uint32_t dilation_h = 1,
      uint32_t dilation_w = 1);


};

} // namespace nn
} // namespace ctl