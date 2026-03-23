

#include "runtime/rt_ir.hpp"
#include "check.hpp"
#include "runtime/rt_attr.hpp"
#include "runtime/rt_op.hpp"
#include "utils/layer_bench.hpp"

#include "ir.h"
#include "log.hpp"
#include "nn/layer.hpp"
#include "nn/layer_factory.hpp"
#include "runtime/rt_param.hpp"
#include "runtime/rt_type.hpp"
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
  std::unordered_map<std::string, decltype(this->operators_.front())>
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


} // namespace ctl
