

#include "nn/param_layer.hpp"
#include "check.hpp"
#include "tensor.hpp"
#include <memory>
#include <vector>

namespace ctl {

ParamLayer::ParamLayer(const std::string &layer_name) : Layer(layer_name) {}

void ParamLayer::InitBiasParam(const uint32_t param_count,
                               const uint32_t param_channel,
                               const uint32_t param_height,
                               const uint32_t param_width) {
  this->bias_.resize(param_count);
  for (uint32_t i = 0; i < param_count; i++) {
    this->bias_[i] =
        std::make_shared<ften>(param_channel, param_height, param_width);
  }
}

void ParamLayer::InitWeightParam(const uint32_t param_count,
                                 const uint32_t param_channel,
                                 const uint32_t param_height,
                                 const uint32_t param_width) {
  this->weights_.resize(param_count);
  for (uint32_t i = 0; i < param_count; i++) {
    this->weights_[i] =
        std::make_shared<ften>(param_channel, param_height, param_width);
  }
}

const std::vector<std::shared_ptr<Tensor<float>>> &ParamLayer::weights() const {
  return this->weights_;
}

const std::vector<std::shared_ptr<Tensor<float>>>& ParamLayer::bias() const {
  return this->bias_;
}


void ParamLayer::set_weight(
    const std::vector<std::shared_ptr<Tensor<float>>> &weights) {
  CHECK_EQ(weights.size(), this->weights_.size());
  
}





} // namespace ctl
