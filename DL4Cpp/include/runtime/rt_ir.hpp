

#ifndef RUNTIME_IR_HPP
#define RUNTIME_IR_HPP

#include "tensor.hpp"
#include <memory>
#include <string>
#include <vector>

namespace ctl {

class RuntimeGraph {

public:
  RuntimeGraph(std::string param_path, std::string bin_path);


  std::vector<ften> get_output(const std::string& output_name , const std::shared_ptr<sften> input_tensors);

  



  void Build();

private:
};

} // namespace ctl

#endif