

#ifndef RELU_HPP

#define RELU_HPP



#include "nn/ops/activation.hpp"

namespace ctl {
namespace nn {




class ReLU : public nn::ActivationLayer {
public:
  explicit ReLU();
  
  
StatusCode Forward(const Tensor& input, Tensor& output) override;

};


} 
}



#endif