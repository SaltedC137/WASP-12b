/**
 *
 *
 *@brief
 *
 */

#ifndef TENSOR_MATH_HPP
#define TENSOR_MATH_HPP

#include "check.hpp"
#include "log.hpp"
#include "tensor.hpp"
#include <armadillo>
#include <format>
#include <memory>
#include <vector>


namespace ctl::math {

  void ElementAdd(const ften &tensor1, const ften &tensor2, ften &output);

  void ElementMultiply(const ften &tensor1, const ften &tensor2, ften &output);

  void Matmul(const ften &tensor1, const ften &tensor2, ften &output);

  void ElementSub(const ften &tensor1, const ften &tensor2, ften &output);

  void ElementDivide(const ften &tensor1, const ften &tensor2, ften &output);

  void AddScalar(const ften &tensor, float scalar, ften &output);

  void MultiplyScalar(const ften &tensor, float scalar, ften &output);

  void ElementExp(const ften &tensor, ften &output);

  void ElementClip(const ften &tensor, float min_val, float max_val,
                   ften &output);

  
  inline sft add(const ften &tensor1, const ften &tensor2) {
    sft output = std::make_shared<ften>(tensor1.shapes());
    ElementAdd(tensor1, tensor2, *output);
    return output;
  }
  

  inline sft sub(const ften &tensor1, const ften &tensor2){
    sft output = std::make_shared<ften>(tensor1.shapes());
    ElementSub(tensor1, tensor2, *output);
    return output;
  }

  
  inline sft mul(const ften &tensor1, const ften &tensor2, ften &output) {
    stf output = std::make_shared<ften>(tensor1.shapes());
    ElementMultiply(tensor1, tensor2, *output);
    return output;
  }

  inline sft div(const ften &tensor1, const ften &tensor2, ften &output){
    sft output = std::make_shared<ften>(tensor1.shapes());
    ElementDivide(tensor1, tensor2, *output);
    return output;
  }

  inline sft matmul(const ften &tensor1, const ften &tensor2, ften &output){
    sft output = std::make_shared<ften>(tensor1.shapes());
    Matmul(tensor1, tensor2, *output);
    return output;
  }
  
  
  
} // namespace ctl::math

#endif
