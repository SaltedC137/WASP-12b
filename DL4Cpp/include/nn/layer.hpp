/**
 * @file layer.hpp
 * @author Aska Lyn
 * @brief
 * @details
 * @date
 */

#ifndef LAYER_HPP
#define LAYER_HPP

#include "core/tensor.hpp"
#include <string>
#include <cstdint>

namespace ctl {


template <typename T> class Layer;

template <> class Layer<int8_t> {}; // int8_t 特化（暂不使用）

template <> class Layer<float> {
public:
  explicit Layer(std::string layer_name) : layer_name_(std::move(layer_name)) {}
  virtual ~Layer() = default;

  virtual std::shared_ptr<Tensor<float>> forward(std::shared_ptr<Tensor<float>> x) = 0;

  const std::string& name() const { return layer_name_; }

protected:
  std::string layer_name_;
  // std::weak_ptr<RuntimeOperator> runtime_operator_;
};



} // namespace ctl

#endif