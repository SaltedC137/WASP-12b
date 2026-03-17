

#ifndef RUNTIME_OPERATOR_HPP
#define RUNTIME_OPERATOR_HPP

#include "runtime/rt_attr.hpp"
#include "runtime/rt_opd.hpp"
#include "runtime/rt_param.hpp"
#include "pnnx/ir.h"
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ctl {

template <typename T> class Layer;

template <typename T>

struct RuntimeOperatorBase {

  // time
  int32_t start_time = -1;

  int32_t end_time = -1;

  int32_t occur_end_time = -1;

  bool forward_only = false;

  // name id
  std::string name;

  std::string type;

  // ptr

  std::shared_ptr<Layer<T>> layer;

  // outputs

  std::vector<std::string> output_names;

  std::shared_ptr<RTOperandBase<T>> output_operand;

  std::map<std::string, std::shared_ptr<RTOperandBase<T>>> output_operators;

  // inputs
  std::map<std::string, std::shared_ptr<RTOperandBase<T>>> input_operands;

  std::vector<std::shared_ptr<RTOperandBase<T>>> input_operands_seq;

  // params
  std::map<std::string, std::shared_ptr<RuntimeParameter>> param;

  // attributes

  std::map<std::string,std::shared_ptr<RuntimeAttribute>> attribute;

  bool has_parameter(const std::string& param_name);

  bool has_attribute(const std::string& attr_name);
};

template<typename T>
bool RuntimeOperatorBase<T>::has_attribute(const std::string& attr_name) {
  return attribute.find(attr_name)!= attribute.end();
}

template<typename T>
bool RuntimeOperatorBase<T>::has_parameter(const std::string& param_name) {
  return param.find(param_name)!= param.end();
}

using RuntimeOperator = RuntimeOperatorBase<float>;
using RuntimeOperatorQuantized = RuntimeOperatorBase<int8_t>;

template<typename T>
class RuntimeOperatorUtils;

template<>
class RuntimeOperatorUtils<float>{
  public:

  static void InitOperatorInput(const std::vector<std::shared_ptr<RuntimeOperator>>& operators);

  static void InitOperatorOutput(const std::vector<pnnx::Operator*>& pnnx_operators, const std::vector<std::shared_ptr<RuntimeOperator>> & operators);
};

} // namespace ctl

#endif