

#include "nn/ops/activation.hpp"
#include "check.hpp"
#include "log.hpp"
#include "ops/sigmoid.hpp"
#include "rt_type.hpp"
#include "tensor.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

StatusCode ActivationLayer::Check(const std::vector<sften> &inputs,
                                  const std::vector<sften> &outputs) {
  const std::string &activation_type = ActivationTypeToString(activation_type_);
  if (inputs.empty()) {
    LOG(ERROR) << "ActivationLayer: " << activation_type
               << " layer has empty inputs.";
    return StatusCode::InferInputsEmpty;
  }
  if (outputs.empty()) {
    LOG(ERROR) << "ActivationLayer: " << activation_type
               << " layer has empty outputs.";
    return StatusCode::InferOutputsEmpty;
  }
  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "ActivationLayer: " << activation_type
               << " layer has different input and output size.";
    return StatusCode::InferDimMismatch;
  }
  return StatusCode::Success;
}

StatusCode
ActivationLayer::Forward(const std::vector<std::shared_ptr<ften>> &inputs,
                         std::vector<std::shared_ptr<ften>> &outputs) {
  StatusCode check_status = Check(inputs, outputs);
  if (check_status != StatusCode::Success) {
    return check_status;
  }
  const uint32_t batch_size = inputs.size();
  const std::string &act_type_str = ActivationTypeToString(activation_type_);
  ActivationFunc act_func = ApplySSEActivation(activation_type_);

#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; i++) {
    const std::shared_ptr<ften> &input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the " + act_type_str +
               " layer has an empty tensor "
        << i << " th";

    std::shared_ptr<ften> output = outputs.at(i);

    if (output == nullptr || output->empty()) {
      output = std::make_shared<ften>(input->shapes());
      outputs.at(i) = output;
    }

    CHECK(output != nullptr && output->shapes() == input->shapes())
        << "The input and output tensor shapes of the " + act_type_str +
               " layer do not match "
        << i << " th";
    act_func(input, output);
  }
  return StatusCode::Success;

}


ActivationLayer::ActivationLayer(nn::ActivationType type,
                                 std::string layer_name)
    : NoneParamLayer(std::move(layer_name)), activation_type_(type) {}


} // namespace nn
} // namespace ctl