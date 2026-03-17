

/**
 * @file rt_op.hpp
 * @author Aska Lyn
 * @brief Runtime operator definitions for computation graph execution
 * @details This header defines the runtime representation of operators (nodes)
 * in a computation graph. Each operator encapsulates layer execution logic,
 * input/output operands, parameters, and attributes. Supports execution order
 * tracking for topological scheduling and data propagation between connected
 * operators. Template-based design enables both float and quantized inference.
 * @date 2026-03-17
 */

#ifndef RUNTIME_OPERATOR_HPP
#define RUNTIME_OPERATOR_HPP

#include "runtime/rt_attr.hpp"
#include "runtime/rt_opd.hpp"
#include "runtime/rt_param.hpp"
#include "pnnx/ir.h"
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ctl {

template <typename T> class Layer;

/**
 * @brief Base template for runtime graph operator
 * @tparam T Data type for operator execution (float, int8_t, etc.)
 * @details Represents a node in the runtime computation graph. Each operator
 * contains execution timing information (for topological ordering), layer
 * pointer for actual computation, input/output operand mappings, and parameter/
 * attribute storage. The design separates graph structure (operands, connections)
 * from execution logic (Layer objects).
 */
template <typename T>
struct RuntimeOperatorBase {
  /// Execution order index (determined by topological sort, -1 if not scheduled)
  int32_t start_time = -1;

  /// End time marker for scheduling (used in dependency tracking)
  int32_t end_time = -1;

  /// Occurrence end time for complex scheduling scenarios
  int32_t occur_end_time = -1;

  /// Forward-only flag indicating single-pass execution constraint
  bool forward_only = false;

  /// Unique operator identifier name within the graph
  std::string name;

  /// Operator type string (e.g., "nn.Conv2d", "nn.ReLU", "pnnx.Input")
  std::string type;

  /// Pointer to the Layer object that performs actual computation
  std::shared_ptr<Layer<T>> layer;

  /// Names of output operators (successor nodes in the graph)
  std::vector<std::string> output_names;

  /// Output operand carrying data to downstream operators
  std::shared_ptr<RTOperandBase<T>> output_operand;

  /// Map of output operators by name (graph edges to successors)
  std::map<std::string, std::shared_ptr<RuntimeOperatorBase<T>>> output_operators;

  /// Map of input operands by producer name (graph edges from predecessors)
  std::map<std::string, std::shared_ptr<RTOperandBase<T>>> input_operands;

  /// Sequential list of input operands (preserves input order for multi-input ops)
  std::vector<std::shared_ptr<RTOperandBase<T>>> input_operands_seq;

  /// Operator parameters (hyperparameters like stride, padding, kernel_size)
  std::map<std::string, std::shared_ptr<RuntimeParameter>> param;

  /// Operator attributes (learnable weights like convolution kernels, biases)
  std::map<std::string, std::shared_ptr<RuntimeAttribute>> attribute;

  /**
   * @brief Check if operator has a specific attribute
   * @param attr_name Attribute name to query
   * @return true if attribute exists, false otherwise
   * @details Performs a lookup in the attribute map. Used during layer
   * initialization to verify required weights are present.
   */
  bool has_attribute(const std::string& attr_name);

  /**
   * @brief Check if operator has a specific parameter
   * @param param_name Parameter name to query
   * @return true if parameter exists, false otherwise
   * @details Performs a lookup in the parameter map. Used during layer
   * initialization to verify required hyperparameters are present.
   */
  bool has_parameter(const std::string& param_name);
};

/**
 * @brief Check if an attribute exists in the operator
 * @tparam T Data type
 * @param attr_name Attribute name to search for
 * @return true if found, false otherwise
 */
template<typename T>
bool RuntimeOperatorBase<T>::has_attribute(const std::string& attr_name) {
  return attribute.find(attr_name)!= attribute.end();
}

/**
 * @brief Check if a parameter exists in the operator
 * @tparam T Data type
 * @param param_name Parameter name to search for
 * @return true if found, false otherwise
 */
template<typename T>
bool RuntimeOperatorBase<T>::has_parameter(const std::string& param_name) {
  return param.find(param_name)!= param.end();
}

/// Float operator type alias (standard precision runtime)
using RuntimeOperator = RuntimeOperatorBase<float>;

/// Int8 operator type alias (quantized runtime)
using RuntimeOperatorQuantized = RuntimeOperatorBase<int8_t>;

/**
 * @brief Utility class for runtime operator initialization
 * @tparam T Data type for operator execution
 * @details Provides static methods for initializing operator input/output
 * tensor spaces before graph execution. Specialized for different data types.
 */
template<typename T>
class RuntimeOperatorUtils;

/**
 * @brief Float specialization of operator utilities
 * @details Handles initialization of input and output tensor spaces for
 * float-precision operators. Manages memory allocation and shape validation.
 */
template<>
class RuntimeOperatorUtils<float>{
  public:
  /**
   * @brief Initialize input tensor spaces for all operators
   * @param operators Vector of operators to initialize
   * @details Allocates input tensor memory based on operand shapes.
   * On first run, creates tensors with proper dimensions. On subsequent
   * runs, validates that input shapes match expected dimensions.
   */
  static void InitOperatorInput(const std::vector<std::shared_ptr<RuntimeOperator>>& operators);

  /**
   * @brief Initialize output tensor spaces for all operators
   * @param pnnx_operators Original PNNX operators (for shape reference)
   * @param operators Runtime operators to initialize
   * @details Allocates output tensor memory based on PNNX graph shape info.
   * Ensures output tensors are properly sized before Forward execution.
   */
  static void InitOperatorOutput(const std::vector<pnnx::Operator*>& pnnx_operators, const std::vector<std::shared_ptr<RuntimeOperator>> & operators);
};

} // namespace ctl

#endif