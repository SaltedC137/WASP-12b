

#ifndef RUNTIME_OPERAND_HPP
#define RUNTIME_OPERAND_HPP

#include "core/tensor.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>
#include "runtime/rt_type.hpp"


namespace ctl {


template <typename T>

struct RTOperandBase {
  explicit RTOperandBase() = default;

  explicit RTOperandBase(std::string name, std::vector<int32_t> shapes,
                         std::vector<std::shared_ptr<Tensor<T>>> data,
                         RuntimeDataType type)
      : name(std::move(name)), shapes(std::move(shapes)), type(type) {}

  explicit RTOperandBase(std::string name, std::vector<int32_t> shapes,
                         uint32_t data_size, RuntimeDataType type)
      : name(std::move(name)), shapes(std::move(shapes)), type(type) {
    datas.resize(data_size);
  }

  size_t size() const;

  std::string name;

  std::vector<int32_t> shapes;

  std::vector<std::shared_ptr<Tensor<T>>> datas;

  RuntimeDataType type = RuntimeDataType::TypeUnknown;
};

template <typename T> size_t RTOperandBase<T>::size() const {
  if (shapes.empty()) {
    return 0;
  }
  size_t size =
      std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
  return size;
}

using RTOperand = RTOperandBase<float>;
using RTOperandU8 = RTOperandBase<int8_t>;

using RTOperandPtr = std::shared_ptr<RTOperand>;

} // namespace ctl
#endif // RUNTIME_OPERAND_HPP