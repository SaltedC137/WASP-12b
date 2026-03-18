

#include "runtime/rt_op.hpp"
#include "core/tensor.hpp"
#include "pnnx/ir.h"
#include "runtime/rt_opd.hpp"
#include "runtime/rt_type.hpp"
#include "utils/check.hpp"
#include "utils/log.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <vector>

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

static void CheckAndReshapeTensor(sften &output_tensor,
                                  const std::vector<int32_t> &operand_shape) {

  CHECK(output_tensor != nullptr) << "Output tensor is null!";
  CHECK(operand_shape.size() >= 2)
      << "Operand shapes length must be at least 2 (e.g., [Batch, Features]).";

  std::vector<uint32_t> target_shape;

  target_shape.reserve(operand_shape.size() - 1);

  for (size_t i = 1; i < operand_shape.size(); ++i) {
    CHECK_GT(operand_shape[i], 0)
        << "Invalid shape dimension at index " << i << ": " << operand_shape[i];
    target_shape.push_back(static_cast<uint32_t>(operand_shape[i]));
  }
  output_tensor->Reshape(target_shape);
}

void RuntimeOperatorUtils<float>::InitOperatorOutput(
    const std::vector<pnnx::Operator *> &pnnx_operators,
    const std::vector<std::shared_ptr<RuntimeOperator>> &operators) {
  CHECK(!pnnx_operators.empty() && !operators.empty() &&
        pnnx_operators.size() == operators.size());
  CHECK(pnnx_operators.size() == operators.size());

  for (uint32_t i = 0; i < pnnx_operators.size(); ++i) {
    const std::vector<pnnx::Operand *> operands = pnnx_operators[i]->outputs;
    if (operands.empty())
      continue;
    if (operands.size() > 1)
      LOG(FATAL) << "Only support one output for each operator yet!";

    pnnx::Operand *operand = operands.front();
    CHECK(operand != nullptr && !operand->shape.empty())
        << "Operand output is null or empty!";
    std::vector<int32_t> operand_shape;
    std::copy_if(operand->shape.begin(), operand->shape.end(),
                 std::back_inserter(operand_shape),
                 [](int32_t dim) { return dim > 0; });

    const auto &runtime_op = operators[i];
    auto &output_tensors = runtime_op->output_operand;
    CHECK((operand_shape.size() == 2 || operand_shape.size() == 4 ||
           operand_shape.size() == 3))
        << "Unsupported output shape size: " << operand_shape.size();

    size_t operand_size = std::accumulate(
        operand_shape.begin(), operand_shape.end(), 1, std::multiplies());

    const int32_t batch = operand_shape[0];
    CHECK_EQ(operand->type, 1) << "The type pnnx should be float32!";
    if (!output_tensors) {
      bool has_found = false;
      for (uint32_t j = 0; j < i; ++j) {
        if (has_found) {
          break;
        }

        const auto &prev_runtime_op = operators.at(j);
        if (!prev_runtime_op->output_operand ||
            prev_runtime_op->occur_end_time != -1) {
          continue;
        }

        if (runtime_op->start_time > prev_runtime_op->occur_end_time) {
          prev_runtime_op->occur_end_time = -1;
        }

        if (runtime_op->start_time > prev_runtime_op->end_time) {
          if (prev_runtime_op->output_operand->size() == operand_size) {
            has_found = true;
            const auto &prev_output_operand = prev_runtime_op->output_operand;

            runtime_op->output_operand = std::make_shared<RuntimeOperand>(
                prev_output_operand->name + "_output", operand_shape, batch,
                RuntimeDataType::TypeFloat32);

            const auto &prev_runtime_op_tensors = prev_output_operand->datas;

            for (uint32_t k = 0; k < batch; ++k) {
              sften prev_output_tensor = prev_runtime_op_tensors.at(k);
              sften output_tensor = std::make_shared<ften>(
                  prev_output_tensor->raw_ptr(), prev_output_tensor->shapes());
              CheckAndReshapeTensor(output_tensor, operand_shape);
              output_tensors->datas[k] = output_tensor;
            }

            prev_runtime_op->occur_end_time = runtime_op->end_time;
          }
        }
      }
      if (!has_found) {
        std::vector<sften> output_operand_datas;
        for (uint32_t k = 0; k < batch; ++k) {
          output_operand_datas.push_back(CreatTensor(operand_shape));
        }
        runtime_op->output_operand = std::make_shared<RuntimeOperand>(
            operand->name + "_output", operand_shape, batch,
            RuntimeDataType::TypeFloat32);
      }
    } else {
      CHECK(batch == output_tensors->datas.size());
      CHECK(output_tensors->type == RuntimeDataType::TypeFloat32);
      CHECK(output_tensors->shapes == operand_shape);
      for (uint32_t k = 0; k < batch; ++k) {
        sften output_tensor = output_tensors->datas[k];
        CheckAndReshapeTensor(output_tensor, operand_shape);
      }
    }
  }
}

} // namespace ctl