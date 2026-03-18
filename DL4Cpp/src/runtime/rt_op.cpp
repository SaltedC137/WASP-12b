

#include "runtime/rt_op.hpp"

#include "check.hpp"
#include "core/tensor.hpp"
#include "core/tensor_math.hpp"
#include "runtime/rt_type.hpp"
#include "utils/log.hpp"
#include <cstdint>
#include <memory>

namespace ctl {

void RuntimeOperatorUtils<float>::InitOperatorInput(
    const std::vector<std::shared_ptr<RuntimeOperator>> &operators) {

  if (operators.empty()) {
    LOG(ERROR) << "No operator to initialize input.";
    return;
  }
  for (auto &op : operators) {
    for (const auto &[_, input_operand] : op->input_operands) {
      if (!input_operand) {
        continue;
      }
      CHECK(input_operand->type == RuntimeDataType::TypeFloat32)
          << "The graph only support float32 yet!";
      const auto &shape = input_operand->shapes;
      CHECK(!shape.empty()) << "Input shape can't be empty!";
      CHECK(shape.size() >= 2 && shape.size() <= 4)
          << "Unsupported input shape size: " << shape.size();
      const int32_t batch = shape.front();
      CHECK(batch > 0) << "Invalid batch size: ";
      auto &input_datas = input_operand->datas;
      if (!input_datas.empty()) {
        CHECK_EQ(input_datas.size(), batch);
      } else {
        input_datas.resize(batch);
      }
    }
  }
}

static sften CreatTensor(const std::vector<int32_t> &operand_shape) {
  if (operand_shape.empty()) {
    LOG(ERROR) << "Operand shape cannot be empty!";
    return nullptr;
  }
  size_t start_idx = (operand_shape.size() >= 2) ? 1 : 0;

  std::vector<uint32_t> tensor_shape;
  tensor_shape.reserve(operand_shape.size() - start_idx);

  for (size_t i = start_idx; i < operand_shape.size(); ++i) {
    if (operand_shape[i] <= 0) {
      LOG(ERROR) << "Invalid shape dimension at index " << i << ": "
                 << operand_shape[i];
      return nullptr;
    }
    tensor_shape.push_back(static_cast<uint32_t>(operand_shape[i]));
  }

  return std::make_shared<ctl::ften>(tensor_shape);
}

void RuntimeOperatorUtils<float>::InitOperatorOutput(
    const std::vector<pnnx::Operator *> &pnnx_operators,
    const std::vector<std::shared_ptr<RuntimeOperator>> &operators) {
  
}



} // namespace ctl