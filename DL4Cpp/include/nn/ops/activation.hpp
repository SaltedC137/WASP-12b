

#ifndef _NN_OPS_ACTIVATION_HPP_
#define _NN_OPS_ACTIVATION_HPP_

#include "core/tensor.hpp"
#include "nn/param_layer.hpp"
#include "runtime/rt_type.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ctl {
namespace nn {

using ActivationFunc = std::function<void(sften, sften)>;

enum ActivationType {
  ActivationType_None = -1,
  ActivationRelu = 0,
  ActivationSilu = 1,
  ActivationSigmoid = 2,
  ActivationHardSwish = 3,
  ActivationHardSigmoid = 4,
  ActivationRelu6 = 5,
};

std::string ActivationTypeToString(ActivationType type);

class ActivationLayer : public NoneParamLayer {
public:
  StatusCode Check(const std::vector<sften> &inputs,
                   const std::vector<sften> &outputs) override;

  StatusCode Forward(const std::vector<std::shared_ptr<ften>> &inputs,
                     std::vector<std::shared_ptr<ften>> outputs) override;

private:
  ActivationType activation_type_ = ActivationType::ActivationType_None;
};

} // namespace activation
} // namespace ctl
#endif