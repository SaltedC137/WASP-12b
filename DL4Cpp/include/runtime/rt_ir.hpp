

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

class RuntimeGraph {

public:
  // Constructor
  RuntimeGraph(std::string param_path, std::string bin_path);

  std::vector<ften> get_output(const std::string &output_name);

  void set_inputs(const std::string &input_name,
                  const std::vector<sften> &inputs);

  bool is_input_op(const std::string &op_name) const;

  bool is_output_op(const std::string &op_name) const;

  void Build();

  void param_path(const std::string &param_path);

  void bin_path(const std::string &bin_path);

  const std::string &param_path() const;

  const std::string &bin_path() const;

  void Forward(bool debug = false);

private:
  bool Init();

  void ReverseToSort();

  template <typename T>
  void
  ReverseToSortInternal(const std::shared_ptr<RuntimeOperatorBase<T>> &root_op,
                        int32_t &current_forward_idx);

  void CreatNodeRelation();

  template <typename T>
  static void InitGraphyOperatorsInput(
      const std::vector<pnnx::Operator *> &input,
      const std::shared_ptr<RuntimeOperatorBase<T>> &runtime_op);

  template <typename T>
  static void InitGraphyOperatorsOutput(
      const std::vector<pnnx::Operator *> &output,
      const std::shared_ptr<RuntimeOperatorBase<T>> &runtime_op);

  template <typename T>
  static void
  InitGraphAttribute(const std::map<std::string, pnnx::Attribute> &attrs,
                     const std::shared_ptr<RuntimeOperatorBase<T>> &runtime_op);

  template <typename T>
  static void
  InitGraphParameter(const std::map<std::string, pnnx::Parameter> &params,
                     const std::shared_ptr<RuntimeOperatorBase<T>> &runtime_op);

  template <typename T>
  std::shared_ptr<Layer<T>>
  CreateLayer(const std::shared_ptr<RuntimeOperatorBase<T>> &runtime_op);

  template <typename T>
  static void
  PropLayerOutputs(const std::shared_ptr<RuntimeOperatorBase<T>> &current_op,
                   const std::vector<std::shared_ptr<Tensor<T>>> &LayerOutputs);

private:
  enum class GraphStatus {

    NeedInit = -2,
    NeedBuild = -1,
    Complete = 0,
  };

private:
  std::string bin_path_;
  std::string param_path_;

  std::unique_ptr<pnnx::Graph> graph_;

  GraphStatus graphstatus_ = GraphStatus::NeedInit;

  std::vector<std::shared_ptr<RuntimeOperator>> input_ops_;
  std::vector<std::shared_ptr<RuntimeOperator>> output_ops_;
  std::vector<std::shared_ptr<RuntimeOperator>> operators_;
};

} // namespace ctl

#endif // RUNTIME_IR_HPP