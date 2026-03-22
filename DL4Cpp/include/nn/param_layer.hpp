

#ifndef PARAM_LAYER_HPP
#define PARAM_LAYER_HPP

#include "nn/layer.hpp"
#include "tensor.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ctl {

class ParamLayer : public Layer<float> {
public:
  explicit ParamLayer(const std::string &layer_name);

  void InitWeightParam(uint32_t param_count, uint32_t param_channel,
                       uint32_t param_height, uint32_t param_width);

  void InitBiasParam(uint32_t param_count, uint32_t param_channel,
                     uint32_t param_height, uint32_t param_width);

  const std::vector<std::shared_ptr<Tensor<float>>> &weights() const override;

  const std::vector<std::shared_ptr<Tensor<float>>> &bias() const override;

  void set_weight(const std::vector<float> &weights) override;

  void set_bias(const std::vector<float> &bias) override;

  void set_weight(
      const std::vector<std::shared_ptr<Tensor<float>>> &weights) override;

  void
  set_bias(const std::vector<std::shared_ptr<Tensor<float>>> &biax) override;

protected:
  std::vector<std::shared_ptr<Tensor<float>>> weights_;
  std::vector<std::shared_ptr<Tensor<float>>> bias_;
};

using NoneParamLayer = Layer<float>;

} // namespace ctl

#endif