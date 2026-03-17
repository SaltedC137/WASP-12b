

#ifndef RUNTIME_IR_HPP
#define RUNTIME_IR_HPP

#include "tensor.hpp"
#include <string>
#include <vector>

namespace ctl {

class RuntimeGraph {

public:
  RuntimeGraph(std::string param_path, std::string bin_path);

  void set_input();

  std::vector<ften> get_output();



  void Build();

private:
};

} // namespace ctl

#endif