/**
 * @author Aska Lyn
 * @brief Learn how to implement tensors in C++
 * @date 2026-03-04 11:10:27
 */


#include "tensor.hpp"
#include <armadillo>
#include <cmath>
#include <cstdint>
#include <vector>

namespace dlc_inf {

Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, channels);
  if (channels == 1 && rows == 1) {
    this->raw_shape_ = std::vector<uint32_t>{cols};
  } else if (channels == 1){
    this->raw_shape_ = std::vector<uint32_t>{rows, cols};
    }else {
    this->raw_shape_ = std::vector<uint32_t>{channels,rows,cols};
    }
}

Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, 1);
  this->raw_shape_ = std::vector<uint32_t>{rows, cols};
}

Tensor<float>::Tensor(uint32_t size) {
  data_ = arma::fcube(1, size, 1);
  this->raw_shape_ = std::vector<uint32_t>{size};
}


    
}