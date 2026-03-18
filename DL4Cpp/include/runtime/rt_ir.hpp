

#ifndef RUNTIME_IR_HPP
#define RUNTIME_IR_HPP

#include "tensor.hpp"
#include <memory>
#include <string>
#include <vector>

namespace ctl {

class RuntimeGraph {

public:


  // Constructor
  RuntimeGraph(std::string param_path, std::string bin_path);


  std::vector<ften> get_output(const std::string& output_name);


  void set_inputs(const std::string& input_name, const std::vector<sften>& inputs);


  bool is_input_op(const std::string& op_name) const;

  bool is_output_op(const std::string& op_name) const;

  void Build();

  void param_path(const std::string& param_path);

  void bin_path(const std::string& bin_path);

  const std::string& param_path() const;

  const std::string& bin_path() const;

  void Forward(bool debug = false);

private:


  void Init();


  std::string param_path_;
  std::string bin_path_;





};

} // namespace ctl

#endif