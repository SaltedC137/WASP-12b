

#include "nn/param_layer.hpp"
#include "core/tensor.hpp"

#include "check.hpp"
#include <cstddef>
#include <cstdint>
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

const std::vector<std::shared_ptr<Tensor<float>>> &ParamLayer::bias() const {
  return this->bias_;
}

void ParamLayer::set_weight(
    const std::vector<std::shared_ptr<Tensor<float>>> &weights) {
  CHECK_EQ(weights.size(), this->weights_.size());
  for (uint32_t i = 0; i < weights.size(); i++) {
    const auto &src_w = weights[i];
    const auto &dst_w = this->weights_[i];
    CHECK(dst_w != nullptr && src_w != nullptr);
    CHECK_EQ(dst_w->rows(), src_w->rows());
    CHECK_EQ(dst_w->cols(), src_w->cols());
    CHECK_EQ(dst_w->channels(), src_w->channels());
  }
  this->weights_ = weights;
}

void ParamLayer::set_bias(
    const std::vector<std::shared_ptr<Tensor<float>>> &bias) {
  if (!this->bias_.empty()) {
    CHECK_EQ(bias.size(), this->bias_.size());
    for (uint32_t i = 0; i < bias.size(); i++) {
      const auto &src_b = bias[i];
      const auto &dst_b = this->bias_[i];
      CHECK(dst_b != nullptr && src_b != nullptr);
      CHECK_EQ(dst_b->rows(), src_b->rows());
      CHECK_EQ(dst_b->cols(), src_b->cols());
      CHECK_EQ(dst_b->channels(), src_b->channels());
    }
    this->bias_ = bias;
  }
}

void ParamLayer::set_weight(const std::vector<float> &weights) {
  const uint32_t batch_size = this->weights_.size();
  CHECK_GT(batch_size, 0);

  size_t weight_size = 0;
  for (const auto &w : this->weights_) {
    weight_size += w->size();
  }
  const size_t elem_size = weights.size();
  CHECK_EQ(weight_size, elem_size);
  CHECK_EQ(elem_size % batch_size, 0);
  const uint32_t blob_size = elem_size / batch_size;
  const float *raw_weights = weights.data();

  for (uint32_t idx = 0; idx < batch_size; ++idx) {
    const uint32_t start_offset = idx * blob_size;
    this->weights_[idx]->Fill(raw_weights + start_offset, blob_size);
  }
}





} // namespace ctl
