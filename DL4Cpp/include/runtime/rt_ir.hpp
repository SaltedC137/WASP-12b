

/**
 * @file rt_ir.hpp
 * @author Aska Lyn
 * @brief Runtime intermediate representation for computation graph execution
 * @details This header defines the runtime computation graph structure that
 * manages the execution of neural network models. The RuntimeGraph class loads
 * model parameters and binary weights from disk, builds the computation graph,
 * and orchestrates forward propagation through topologically sorted operators.
 * Supports debug mode for execution tracing and provides input/output tensor
 * management for model inference.
 * @date 2026-03-17
 */

#ifndef RUNTIME_IR_HPP
#define RUNTIME_IR_HPP

#include "ir.h"
#include "rt_op.hpp"
#include "tensor.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ctl {

/**
 * @brief Runtime computation graph manager
 * @details Manages the lifecycle of a computation graph from loading to
 * execution. The graph is built from PNNX intermediate representation, with
 * support for topological sorting, operator initialization, and forward
 * propagation. Maintains references to input/output operators and all
 * intermediate operators for efficient traversal during inference.
 */
class RuntimeGraph {

public:
  /**
   * @brief Construct a new Runtime Graph object
   * @param param_path Path to the PNNX parameter file (.param)
   * @param bin_path Path to the binary weights file (.bin)
   * @details Initializes the graph with file paths. The graph is not built
   * until Build() is called, allowing for deferred initialization.
   */
  RuntimeGraph(std::string param_path, std::string bin_path);

  /**
   * @brief Get output tensor by name
   * @param output_name Name of the output operator to retrieve
   * @return std::vector<ften> Vector of output tensors
   * @details Retrieves the output tensor(s) from a specified operator in the
   * graph. Used after Forward() to access inference results.
   */
  std::vector<sften> get_outputs(const std::string &output_name);

  /**
   * @brief Set input tensor for a named input operator
   * @param input_name Name of the input operator
   * @param inputs Vector of input tensors to set
   * @details Binds input data to the graph's input operators before Forward().
   * The input tensors must match the expected shape defined in the model.
   */
  void set_inputs(const std::string &input_name,
                  const std::vector<sften> &inputs);

  /**
   * @brief Check if an operator is an input node
   * @param op_name Operator name to check
   * @return true if the operator is an input node, false otherwise
   * @details Input nodes are graph entry points (typically pnnx.Input ops).
   * Used to identify where input data should be fed into the graph.
   */
  bool is_input_op(const std::string &op_name) const;

  /**
   * @brief Check if an operator is an output node
   * @param op_name Operator name to check
   * @return true if the operator is an output node, false otherwise
   * @details Output nodes are graph exit points where results are collected.
   * Used to identify which operators produce the final inference outputs.
   */
  bool is_output_op(const std::string &op_name) const;

  /**
   * @brief Build the computation graph from parameter and binary files
   * @details Parses the PNNX parameter file to construct the graph structure,
   * loads binary weights, initializes operators with their parameters and
   * attributes, and performs topological sorting for execution ordering.
   * Must be called before Forward().
   */
  void Build();

  /**
   * @brief Set the parameter file path
   * @param param_path New path to the parameter file
   * @details Updates the parameter file path. Should be called before Build().
   * Changing this after Build() requires re-initialization.
   */
  void set_param_path(const std::string &param_path);

  /**
   * @brief Set the binary weights file path
   * @param bin_path New path to the binary weights file
   * @details Updates the binary file path. Should be called before Build().
   * Changing this after Build() requires re-initialization.
   */
  void set_bin_path(const std::string &bin_path);

  /**
   * @brief Get the parameter file path
   * @return const std::string& Reference to the parameter file path
   * @details Returns the current parameter file path used by the graph.
   */
  const std::string &param_path() const;

  /**
   * @brief Get the binary weights file path
   * @return const std::string& Reference to the binary file path
   * @details Returns the current binary file path used by the graph.
   */
  const std::string &bin_path() const;

  /**
   * @brief Execute forward propagation through the graph
   * @param debug If true, enable debug output for execution tracing
   * @details Runs the computation graph in topological order, executing each
   * operator's layer and propagating outputs to downstream operators. Debug
   * mode prints execution details for each operator for troubleshooting.
   */
  void Forward(bool debug = false);

private:
  /**
   * @brief Initialize the graph from loaded files
   * @return true if initialization succeeds, false otherwise
   * @details Internal initialization routine that loads the PNNX graph,
   * creates runtime operators, and prepares the graph for building.
   * Called internally by Build().
   */
  bool Init();

  /**
   * @brief Perform reverse topological sort to determine execution order
   * @details Traverses the graph from output nodes backwards to assign
   * execution indices to operators. Ensures operators are executed in
   * valid topological order (all dependencies computed first).
   */
  void ReverseToSort();

  /**
   * @brief Internal recursive function for reverse topological sorting
   * @tparam T Data type for operator execution
   * @param root_op Current operator being visited in traversal
   * @param current_forward_idx Reference to the current execution index counter
   * @details Recursively traverses the graph backwards from outputs, assigning
   * execution indices. Uses depth-first search with visited tracking to ensure
   * each operator is processed exactly once.
   */
  template <typename T>
  void
  ReverseToSortInternal(const std::shared_ptr<RuntimeOperatorBase<T>> &root_op,
                        int32_t &current_forward_idx);

  /**
   * @brief Create parent-child relationships between operators
   * @details Establishes the operator connectivity based on PNNX graph edges.
   * Links input/output operands between connected operators to enable data
   * flow during forward propagation.
   */
  void CreateNodeRelation();

  /**
   * @brief Initialize input operands for graph operators
   * @tparam T Data type for operator execution
   * @param input Vector of PNNX input operands
   * @param runtime_op Corresponding runtime operator to initialize
   * @details Sets up the input operand structure for runtime operators based
   * on PNNX graph input definitions.
   */
  template <typename T>
  static void InitGraphOperatorsInput(
      const std::vector<pnnx::Operand *> &input,
      const std::shared_ptr<RuntimeOperatorBase<T>> &runtime_op);

  /**
   * @brief Initialize output operands for graph operators
   * @tparam T Data type for operator execution
   * @param output Vector of PNNX output operands
   * @param runtime_op Corresponding runtime operator to initialize
   * @details Sets up the output operand structure for runtime operators based
   * on PNNX graph output definitions.
   */
  template <typename T>
  static void InitGraphOperatorsOutput(
      const std::vector<pnnx::Operand *> &output,
      const std::shared_ptr<RuntimeOperatorBase<T>> &runtime_op);

  /**
   * @brief Initialize operator attributes from PNNX graph
   * @tparam T Data type for operator execution
   * @param attrs Map of PNNX attributes to copy
   * @param runtime_op Runtime operator to initialize with attributes
   * @details Copies learnable attributes (weights, biases) from PNNX graph
   * to runtime operator. Attributes are stored as RuntimeAttribute objects
   * for type-safe access during layer execution.
   */
  template <typename T>
  static void
  InitGraphAttribute(const std::map<std::string, pnnx::Attribute> &attrs,
                     const std::shared_ptr<RuntimeOperatorBase<T>> &runtime_op);

  /**
   * @brief Initialize operator parameters from PNNX graph
   * @tparam T Data type for operator execution
   * @param params Map of PNNX parameters to copy
   * @param runtime_op Runtime operator to initialize with parameters
   * @details Copies hyperparameters (stride, padding, kernel_size, etc.) from
   * PNNX graph to runtime operator. Parameters are stored as RuntimeParameter
   * objects for type-safe access during layer execution.
   */
  template <typename T>
  static void
  InitGraphParameter(const std::map<std::string, pnnx::Parameter> &params,
                     const std::shared_ptr<RuntimeOperatorBase<T>> &runtime_op);

  /**
   * @brief Create a Layer object for the given operator
   * @tparam T Data type for operator execution
   * @param runtime_op Runtime operator to create layer for
   * @return std::shared_ptr<Layer<T>> Created layer object
   * @details Instantiates the appropriate Layer subclass based on the operator
   * type (e.g., Conv2dLayer for nn.Conv2d). The layer encapsulates the actual
   * computation logic for the operator.
   */
  template <typename T>
  std::shared_ptr<Layer<T>>
  CreateLayer(const std::shared_ptr<RuntimeOperatorBase<T>> &runtime_op);

  /**
   * @brief Propagate layer outputs to downstream operators
   * @tparam T Data type for operator execution
   * @param current_op Current operator whose outputs are being propagated
   * @param LayerOutputs Vector of output tensors from the layer
   * @details Distributes the output tensors from a completed layer to all
   * successor operators in the graph. This establishes the data flow between
   * operators during forward propagation.
   */
  template <typename T>
  static void
  PropLayerOutputs(const std::shared_ptr<RuntimeOperatorBase<T>> &current_op,
                   const std::vector<std::shared_ptr<Tensor<T>>> &LayerOutputs);

private:
  /**
   * @brief Graph initialization status enumeration
   * @details Tracks the current state of the graph to ensure proper
   * initialization sequence. Prevents invalid operations on uninitialized
   * or partially built graphs.
   */
  enum class GraphStatus {
    NeedInit = -2,  ///< Graph needs initialization (files not loaded)
    NeedBuild = -1, ///< Graph initialized but not yet built
    Complete = 0,   ///< Graph fully built and ready for execution
  };

public:
  GraphStatus graph_status() const;


private:
  std::string bin_path_;   ///< Path to binary weights file
  std::string param_path_; ///< Path to PNNX parameter file

  std::unique_ptr<pnnx::Graph> graph_; ///< PNNX graph structure

  GraphStatus graphstatus_ = GraphStatus::NeedInit; ///< Current graph status

  std::vector<std::shared_ptr<RuntimeOperator>> input_ops_; ///< Input operators
  std::vector<std::shared_ptr<RuntimeOperator>>
      output_ops_; ///< Output operators
  std::vector<std::shared_ptr<RuntimeOperator>> operators_; ///< All operators
};

} // namespace ctl

#endif // RUNTIME_IR_HPP