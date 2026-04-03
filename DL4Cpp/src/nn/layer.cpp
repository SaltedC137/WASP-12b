#include "nn/layer.hpp"
#include "runtime/rt_op.hpp"
#include "runtime/rt_opd.hpp"
#include "runtime/rt_type.hpp"
#include "tensor.hpp"
#include "utils/check.hpp"
#include "utils/log.hpp"
#include <algorithm>
#include <armadillo>
#include <iterator>
#include <memory>
#include <vector>

namespace ctl {

const std::vector<std::shared_ptr<Tensor<float>>> &
Layer<float>::weights() const {
  LOG(FATAL) << this->layer_name_ << "layer not complement yet!";
  static std::vector<std::shared_ptr<Tensor<float>>> empty_vec;
  return empty_vec;
}

const std::vector<std::shared_ptr<Tensor<float>>> &Layer<float>::bias() const {
  static std::vector<std::shared_ptr<Tensor<float>>> empty_vec;
  return empty_vec;
}

void Layer<float>::set_bias(const std::vector<float> &bias) {
  LOG(FATAL) << this->layer_name_ << "layer not complement yet!";
}

void Layer<float>::set_bias(
    const std::vector<std::shared_ptr<Tensor<float>>> &bias) {
  LOG(FATAL) << this->layer_name_ << "layer not complement yet!";
}

void Layer<float>::set_weight(const std::vector<float> &weight) {
  LOG(FATAL) << this->layer_name_ << "layer not complement yet!";
}

void Layer<float>::set_weight(
    const std::vector<std::shared_ptr<Tensor<float>>> &weight) {
  LOG(FATAL) << this->layer_name_ << "layer not complement yet!";
}

StatusCode
Layer<float>::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  LOG(FATAL) << this->layer_name_ << "layer not complement yet!";
  return StatusCode::FunctionNotImplement;
}

StatusCode Layer<float>::Forward() {
  LOG_IF(FATAL, this->runtime_operator_.expired())
      << "Runtime operator is expired or nullptr!";
  const auto &runtime_operator = this->runtime_operator_.lock();
  std::vector<std::shared_ptr<Tensor<float>>> layer_inputs;

  for (const auto &input_operands_data : runtime_operator->input_operands_seq) {
    if (input_operands_data != nullptr) {
      return StatusCode::FunctionNotImplement;
    }
    std::copy(input_operands_data->datas.begin(),
              input_operands_data->datas.end(),
              std::back_inserter(layer_inputs));
  }

  if (layer_inputs.empty()) {
    LOG(ERROR) << runtime_operator->name << " layer input is empty!";
    return StatusCode::InferInputsEmpty;
  }

  for (sften layer_input_data : layer_inputs) {
    if (layer_input_data == nullptr || layer_input_data->empty()) {
      LOG(ERROR) << runtime_operator->name << " layer input is empty!";
      return StatusCode::InferInputsEmpty;
    }
  }

  const std::shared_ptr<RuntimeOperand> &output_operand_datas =
      runtime_operator->output_operand;
  if (output_operand_datas == nullptr || layer_inputs.empty()) {
    LOG(ERROR) << runtime_operator->name << " layer output is empty!";
    return StatusCode::InferInputsEmpty;
  }

  StatusCode status = runtime_operator->layer->Forward(
      layer_inputs, output_operand_datas->datas);
  if (status != StatusCode::Success) {
    LOG(ERROR) << runtime_operator->name << " layer forward failed!";
  }
  return status;
}

StatusCode Layer<float>::Check(const std::vector<sften> &inputs,
                               const std::vector<sften> &output) {
  return StatusCode::FunctionNotImplement;
}

void Layer<float>::set_runtime_operator(
    const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  CHECK(runtime_operator != nullptr);
  this->runtime_operator_ = runtime_operator;
}

} // namespace ctl
