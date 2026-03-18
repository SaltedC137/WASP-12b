

#include "runtime/rt_op.hpp"

#include "core/tensor.hpp"
#include "core/tensor_math.hpp"
#include "utils/log.hpp"
#include <memory>

namespace ctl {


void RuntimeOperatorUtils<float>::InitOperatorInput(
    const std::vector<std::shared_ptr<RuntimeOperator>> &operators) {

  if (operators.empty()) {
    LOG(ERROR) << "No operator to initialize input.";
    return;
  }
  for (auto &op : operators) {
    if (op->input_operands.empty()) {
      continue;
    }else {
    const std::map<std::string,std::shared_ptr<RuntimeOperand>>& )
    }
  }
}

}