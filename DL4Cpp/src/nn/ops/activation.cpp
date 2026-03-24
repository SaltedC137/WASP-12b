

#include "nn/ops/activation.hpp"
#include <string>

namespace ctl {
namespace nn {

std::string ActivationTypeToString(ActivationType type) {
  std::string activation_type_str;
  switch (type) {
  case ActivationType::ActivationType_None: {
    activation_type_str = "Unknown";
    break;
  }
  case ActivationType::ActivationRelu: {
    activation_type_str = "ReLU";
    break;
  }
  case ActivationType::ActivationSilu: {
    activation_type_str = "SiLU";
    break;
  }
  case ActivationType::ActivationSigmoid: {
    activation_type_str = "Sigmoid";
    break;
  }
  case ActivationType::ActivationHardSwish: {
    activation_type_str = "HardSwish";
    break;
  }
  case ActivationType::ActivationHardSigmoid: {
    activation_type_str = "HardSigmoid";
    break;
  }
  case ActivationType::ActivationRelu6: {
    activation_type_str = "ReLU6";
    break;
  }
  default: {
    activation_type_str = "Unknown";
  }
  }
  return activation_type_str;
}

} // namespace activation
} // namespace ctl