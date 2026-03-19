

#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"
#include <cstdint>
#include "runtime/rt_op.hpp"



namespace ctl {

template <typename T> class Layer;

template <> class Layer<int8_t> {};

template <> class Layer<float> {

public:
  explicit Layer(std::string layer_name) : layer_name_(std::move(layer_name)) {}





 protected:
  std::string layer_name_;
  std::weak_ptr<RuntimeOperator> runtime_operator_;


};

} // namespace ctl

#endif