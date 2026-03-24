

#include "runtime/rt_ir.hpp"
#include "check.hpp"
#include "runtime/rt_attr.hpp"
#include "runtime/rt_op.hpp"
#include "tensor.hpp"
#include "utils/layer_bench.hpp"

#include "ir.h"
#include "log.hpp"
#include "nn/layer.hpp"
#include "nn/layer_factory.hpp"
#include "runtime/rt_param.hpp"
#include "runtime/rt_type.hpp"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ctl {

RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
    : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {}

void RuntimeGraph::set_bin_path(const std::string &bin_path) {
  this->bin_path_ = bin_path;
}

void RuntimeGraph::set_param_path(const std::string &param_path) {
  this->param_path_ = param_path;
}

const std::string &RuntimeGraph::param_path() const {
  return this->param_path_;
}

const std::string &RuntimeGraph::bin_path() const { return this->bin_path_; }

static bool IsQuantizedOp(const pnnx::Operator *op) { return false; }

RuntimeGraph::GraphStatus RuntimeGraph::graph_status() const {
  return this->graphstatus_;
}

bool RuntimeGraph::Init() {
  if (this->bin_path_.empty() || this->param_path_.empty()) {
    LOG(ERROR) << "The bin path or param path is empty";
    return false;
  }

  this->graph_ = std::make_unique<pnnx::Graph>();
  if (this->graph_->load(param_path_, bin_path_) != 0) {
    LOG(ERROR) << "Failed to load graph from " << param_path_ << " and "
               << bin_path_;
    return false;
  }

  const std::vector<pnnx::Operator *> &pnnx_ops = this->graph_->ops;
  if (pnnx_ops.empty()) {
    LOG(ERROR) << "Can not read the layer' define (empty operator list)";
    return false;
  }

  this->operators_.clear();
  this->operators_.reserve(pnnx_ops.size());

  for (const pnnx::Operator *op : pnnx_ops) {
    if (!op) {
      LOG(WARNING) << "Meet an empty node in the model, skipping.";
      continue;
    }
    if (IsQuantizedOp(op)) {
      LOG(ERROR) << "Unsupported quantize operator in the model:" << op->name
                 << "type:" << op->type << " <- is not supported";
      return false;
    }

    auto runtime_op = std::make_shared<RuntimeOperator>();
    runtime_op->name = op->name;
    runtime_op->type = op->type;

    InitGraphOperatorsInput(op->inputs, runtime_op);
    InitGraphOperatorsOutput(op->outputs, runtime_op);

    InitGraphAttribute(op->attrs, runtime_op);
    InitGraphParameter(op->params, runtime_op);

    this->operators_.push_back(runtime_op);
  }
  this->graphstatus_ = GraphStatus::NeedBuild;
  return true;
}

template <typename T>
void RuntimeGraph::InitGraphOperatorsInput(
    const std::vector<pnnx::Operand *> &input,
    const std::shared_ptr<RuntimeOperatorBase<T>> &runtime_op) {
  if (input.empty()) {
    LOG(ERROR) << "The operator input is empty";
    return;
  }
  CHECK(runtime_op != nullptr) << "The runtime_op is null";
  for (const pnnx::Operand *input_ : input) {
    if (!input_) {
      continue;
    }

    std::vector<int32_t> dims;
    const pnnx::Operator *producer = input_->producer;

    for (int32_t dim : input_->shape) {
      dims.push_back(dim);
    }
    CHECK(!dims.empty()) << "Input shape is empty";

    auto runtime_operand = std::make_shared<RuntimeOperandBase<T>>();
    runtime_operand->name = input_->name;
    runtime_operand->shapes = dims;

    runtime_op->input_operands.insert({producer->name, runtime_operand});
    runtime_op->input_operands_seq.push_back(runtime_operand);

    switch (input_->type) {
    case 1: {
      runtime_operand->type = RuntimeDataType::TypeFloat32;
      break;
    }
    case 7: {
      runtime_operand->type = RuntimeDataType::TypeInt8;
      break;
    }
    default: {
      LOG(FATAL) << "Unsupported input type: " << input_->type;
    }
    }
  }
}

template <typename T>

void RuntimeGraph::InitGraphParameter(
    const std::map<std::string, pnnx::Parameter> &params,
    const std::shared_ptr<RuntimeOperatorBase<T>> &runtime_op) {
  if (params.empty())
    return;

  CHECK(runtime_op != nullptr) << "The runtime_op is null";

  for (const auto &[name, parameter] : params) {
    std::shared_ptr<RuntimeParameter> runtime_param = nullptr;
    switch (parameter.type) {
    case int32_t(RuntimeParameterType::ParameterUnknown): {
      runtime_param = std::make_shared<RuntimeParameter>();
      break;
    }
    case int32_t(RuntimeParameterType::ParameterBool): {
      runtime_param = std::make_shared<RuntimeParameterBool>(parameter.b);
      break;
    }
    case int32_t(RuntimeParameterType::ParameterInt): {
      runtime_param = std::make_shared<RuntimeParameterInt>(parameter.i);
      break;
    }
    case int32_t(RuntimeParameterType::ParameterFloat): {
      runtime_param = std::make_shared<RuntimeParameterFloat>(parameter.f);
      break;
    }
    case int32_t(RuntimeParameterType::ParameterString): {
      runtime_param = std::make_shared<RuntimeParameterString>(parameter.s);
      break;
    }
    case int32_t(RuntimeParameterType::ParameterIntArray): {
      runtime_param = std::make_shared<RuntimeParameterIntArray>(parameter.ai);
      break;
    }
    case int32_t(RuntimeParameterType::ParameterFloatArray): {
      runtime_param =
          std::make_shared<RuntimeParameterFloatArray>(parameter.af);

      break;
    }
    case int32_t(RuntimeParameterType::ParameterStringArray): {
      runtime_param =
          std::make_shared<RuntimeParameterStringArray>(parameter.as);
      break;
    }
    default: {
      LOG(FATAL) << "Unsupported parameter type: " << parameter.type;
    }
    }
  }
}

template <typename T>
void RuntimeGraph::InitGraphAttribute(
    const std::map<std::string, pnnx::Attribute> &attrs,
    const std::shared_ptr<RuntimeOperatorBase<T>> &runtime_op) {
  if (attrs.empty())
    return;
  CHECK(runtime_op != nullptr) << "The runtime_op is null";
  for (const auto &[name, attribute] : attrs) {
    switch (attribute.type) {
    case 1: {
      std::shared_ptr<RuntimeAttribute> runtime_arrtribute =
          std::make_shared<RuntimeAttribute>(
              attribute.shape, RuntimeDataType::TypeFloat32, attribute.data);
      runtime_op->attribute.insert({name, runtime_arrtribute});
      break;
    }
    default: {
      LOG(FATAL) << "Unsupported attribute type: " << attribute.type;
    }
    }
  }
}

void RuntimeGraph::Build() {
  if (graphstatus_ == GraphStatus::Complete) {
    LOG(INFO) << "The graph has already been built";
    return;
  }

  if (graphstatus_ == GraphStatus::NeedInit) {
    bool init_graph = Init();
    LOG_IF(FATAL, !init_graph || graphstatus_ == GraphStatus::NeedInit)
        << "Failed to initialize the graph";
  }

  CHECK(graphstatus_ >= GraphStatus::NeedBuild)
      << "Graph status is not valid" << int32_t(graphstatus_);

  LOG_IF(FATAL, this->operators_.empty()) << "The operator list is empty";

  // TODO: build the graph
  CreateNodeRelation();

  // TODO: sort the graph
  ReverseToSort();

  RuntimeOperatorUtils<float>::InitOperatorInput(operators_);
  RuntimeOperatorUtils<float>::InitOperatorOutput(graph_->ops, operators_);

  graphstatus_ = GraphStatus::Complete;
  if (graph_ != nullptr) {
    graph_.reset();
    graph_ = nullptr;
  }
}

template <typename T>
StatusCode ExcuteLayer(const std::shared_ptr<Layer<T>> &layer,
                       const std::string &op_name, const std::string &op_type,
                       bool is_debug) {
  CHECK(layer != nullptr) << "The layer is null";
  StatusCode status;
  if (is_debug) {
    bench::LayerTimeLogging layer_time_logging(op_name, op_type);
    status = layer->Forward(); ///?????????
  } else {
    status = layer->Forward(); ///?????????
  }
  return status;
}

void RuntimeGraph::Forward(bool debug) {
  if (graphstatus_ == GraphStatus::Complete) {
    LOG(FATAL) << "Graph need be build!" << ", current state is "
               << int32_t(graphstatus_);
  }

  if (debug) {
    bench::LayerTimeStatsSingleton::ClearTimeStats();
  }

  for (const auto &current_op : operators_) {
    current_op->has_forward = false;
    CHECK_GT(current_op->start_time, 0);

    if (is_input_op(current_op->name) || is_output_op(current_op->name)) {
      current_op->has_forward = true;
      continue;
    }

    CHECK(current_op->layer != nullptr)
        << "The layer corresponding to the op " << current_op->name
        << " is empty, indicating that it may not have been created.";

    StatusCode status = ExcuteLayer(current_op->layer, current_op->name,
                                    current_op->type, debug);

    CHECK(status == StatusCode::Success)
        << current_op->layer->layer_name() // ??????????
        << " layer forward failed, error code: " << int32_t(status);

    current_op->has_forward = true;
    PropLayerOutputs(current_op, current_op->output_operand->datas);
    if (debug) {
      bench::LayerTimeLogging::SummaryLogging();
    }

    for (const auto &op : operators_) {
      LOG_IF(FATAL, !op->has_forward)
          << "The operator: " << op->name << " has not been forward yet!";
    }
  }
}

template <typename T>
std::shared_ptr<Layer<T>>
RuntimeGraph::CreateLayer(const std::shared_ptr<RuntimeOperatorBase<T>> &op) {
  LOG_IF(FATAL, !op) << "Operator is empty!";
  auto layer = LayerRegister::CreateLayer(op);
  LOG_IF(FATAL, !layer) << "Layer init failed " << op->type;
  return layer;
}

void RuntimeGraph::CreateNodeRelation() {
  std::unordered_map<std::string, std::shared_ptr<ctl::RuntimeOperatorBase<float>>>
      op_hash_map;
  op_hash_map.reserve(this->operators_.size());
  for (const auto &op : this->operators_) {
    op_hash_map[op->name] = op;
  }

  for (const auto &op : this->operators_) {
    for (const auto &next_op_name : op->output_names) {
      auto it = op_hash_map.find(next_op_name);
      if (it != op_hash_map.end() && it->second != op) {
        op->output_operators.insert({next_op_name, it->second});
      } else {
        LOG(WARNING) << "Graph relation warning: Cannot find output operator ["
                     << next_op_name << "] for node [" << op->name << "]";
      }
    }
    if (op->type == "pnnx.Input" || op->type == "pnnx.Output") {
      continue;
    }

    auto layer = RuntimeGraph::CreateLayer(op);
    CHECK(layer != nullptr) << "Create layer failed for op " << op->name;
    op->layer = layer;
    layer->set_runtime_operator(op);
  }
}

void RuntimeGraph::set_inputs(const std::string &input_name,
                              const std::vector<sften> &inputs) {
  CHECK(this->graphstatus_ == GraphStatus::Complete) << "Graph need be build!";
  std::shared_ptr<RuntimeOperator> input_op;
  for (auto op : this->input_ops_) {
    if (op->name == input_name) {
      input_op = op;
      break;
    }
  }
  CHECK(input_op != nullptr) << "Cannot find input operator " << input_name;
  PropLayerOutputs(input_op, inputs);
}

template <typename T>
void RuntimeGraph::PropLayerOutputs(
    const std::shared_ptr<RuntimeOperatorBase<T>> &current_op,
    const std::vector<std::shared_ptr<Tensor<T>>> &LayerOutputs) {
  for (const auto &[_, output_op] : current_op->output_operators) {
    const auto &next_input_operands = output_op->input_operands;
    const auto &next_input_operands_iter =
        next_input_operands.find(current_op->name);
    if (next_input_operands_iter != next_input_operands.end()) {
      std::vector<sten<T>> &next_input_datas =
          next_input_operands_iter->second->datas;
      for (uint32_t i = 0; i < LayerOutputs.size(); i++) {
        const sten<T> &layer_output_data = LayerOutputs.at(i);
        if (next_input_datas.at(i) != nullptr) {
          CHECK(next_input_datas.at(i)->shapes() ==
                layer_output_data->shapes());
        }
        next_input_datas.at(i) = layer_output_data;
      }
    }
  }
}

void RuntimeGraph::ReverseToSort() {
  for (const auto &op : operators_) {
    if (op != nullptr && !op->has_forward) {
      int32_t current_forward_idx = 0;
      this->ReverseToSortInternal(op, current_forward_idx);
    }
  }

  // lambda function to sort the operators by start_time

  std::sort(operators_.begin(), operators_.end(),
            [](const auto &op1, const auto &op2) {
              return op1->start_time < op2->start_time;
            });

  int32_t forward_index = 1;
  for (const auto &op : operators_) {
    op->start_time = forward_index;
    forward_index++;
  }

  for (const auto &op : operators_) {
    const auto &next_ops = op->output_operators;
    int32_t last_forward_index = -1;
    for (const auto &[_, next_op] : next_ops) {
      if (next_op->start_time >= last_forward_index) {
        last_forward_index = next_op->start_time;
      }
    }
    if (last_forward_index == -1) {
      op->end_time = op->start_time + 1;
    } else {
      op->end_time = last_forward_index;
    }
    op->occur_end_time = -1;
  }
}

template <typename T>
void RuntimeGraph::ReverseToSortInternal(
    const std::shared_ptr<RuntimeOperatorBase<T>> &root_op,
    int32_t &current_forward_idx) {
  LOG_IF(FATAL, !root_op) << "The root_op is null";
  if (root_op->input_operands.empty() && !root_op->has_forward) {
    this->input_ops_.push_back(root_op);
  }
  if (root_op->output_names.empty() && !root_op->has_forward) {
    this->output_ops_.push_back(root_op);
  }

  root_op->has_forward = true;
  const auto &next_ops = root_op->output_operators;
  for (const auto &[_, op] : next_ops) {
    if (op != nullptr && !op->has_forward) {
      this->ReverseToSortInternal(op, current_forward_idx);
    }
  }

  for(const auto&[_,op]:next_ops){
    CHECK_EQ(op->has_forward, true);
  }
  root_op->start_time = current_forward_idx;
  current_forward_idx++;
}


std::vector<sften> RuntimeGraph::get_outputs(const std::string& output_name)  {
  CHECK(this->graphstatus_ == GraphStatus::Complete);
  std::shared_ptr<RuntimeOperator> output_op;
  for (auto op : this->output_ops_) {
    if (op->name == output_name) {
      output_op = op;
    }
  }

  CHECK(output_op != nullptr) << "Can not find the output operator: " << output_name;
  std::vector<sften> outputs;
  for (const auto& input_operand : output_op->input_operands_seq) {
    std::copy(input_operand->datas.begin(), input_operand->datas.end(),
              std::back_inserter(outputs));
  }
  return outputs;
}


bool RuntimeGraph::is_input_op(const std::string& op_name) const {
  for (auto op : this->input_ops_) {
    CHECK(op != nullptr);
    if (op->name == op_name) {
      return true;
    }
  }
  return false;
}

bool RuntimeGraph::is_output_op(const std::string& op_name) const {
  for (auto op : this->output_ops_) {
    CHECK(op != nullptr);
    if (op->name == op_name) {
      return true;
    }
  }
  return false;
}

template <typename T>
void RuntimeGraph::InitGraphOperatorsOutput(
    const std::vector<pnnx::Operand*>& outputs,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
  if (outputs.empty()) {
    return;
  }
  CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
  for (const pnnx::Operand* output : outputs) {
    if (!output) {
      continue;
    }
    const auto& consumers = output->consumers;
    for (const auto& c : consumers) {
      runtime_operator->output_names.push_back(c->name);
    }
  }
}

} // namespace ctl
