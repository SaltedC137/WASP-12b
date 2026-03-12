/**
 * @author Aska Lyn
 * @brief Learn how to implement tensors in C++
 * @date 2026-03-04 11:10:27
 */

#include "tensor.hpp"
#include "check.hpp"
#include "log.hpp"

#include <algorithm>
#include <armadillo>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>

namespace dlc_inf {

// Construct 3D tensor with specified channels, rows, and cols
Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, channels);
  if (channels == 1 && rows == 1) {
    this->raw_shape_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shape_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shape_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

// Construct 2D tensor (matrix) with rows and cols
Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, 1);
  this->raw_shape_ = std::vector<uint32_t>{rows, cols};
}

// Construct 1D tensor (vector) with specified size
Tensor<float>::Tensor(uint32_t size) {
  data_ = arma::fcube(1, size, 1);
  this->raw_shape_ = std::vector<uint32_t>{size};
}

// Construct tensor from shape vector
Tensor<float>::Tensor(const std::vector<uint32_t> &shapes) {
  CHECK(!shapes.empty() && shapes.size() <= 3)
      << "Shape vector cannot be empty";
  const uint32_t remaining = 3 - shapes.size();
  std::vector<uint32_t> shapes_(3, 1);
  std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

  const uint32_t channels = shapes_.at(0);
  const uint32_t rows = shapes_.at(1);
  const uint32_t cols = shapes_.at(2);

  data_ = arma::fcube(rows, cols, channels);

  if (channels == 1 && rows == 1) {
    this->raw_shape_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shape_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shape_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

// Copy constructor - deep copy
Tensor<float>::Tensor(const Tensor &tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shape_ = tensor.raw_shape_;
  }
}

// Move constructor - transfer ownership
Tensor<float>::Tensor(Tensor<float> &&tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shape_ = tensor.raw_shape_;
  }
}

// Move assignment operator
Tensor<float> &Tensor<float>::operator=(Tensor<float> &&tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shape_ = tensor.raw_shape_;
  }
  return *this;
}

// Copy assignment operator
Tensor<float> &Tensor<float>::operator=(const Tensor &tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shape_ = tensor.raw_shape_;
  }
  return *this;
}

// Get number of rows
uint32_t Tensor<float>::rows() const {
  CHECK(!this->data_.empty());
  return this->data_.n_rows;
}

// Get number of columns
uint32_t Tensor<float>::cols() const {
  CHECK(!this->data_.empty());
  return this->data_.n_cols;
}

// Get number of channels
uint32_t Tensor<float>::channels() const {
  CHECK(!this->data_.empty());
  return this->data_.n_slices;
}

// Get total number of elements
uint32_t Tensor<float>::size() const {
  CHECK(!this->data_.empty());
  return this->data_.size();
}

// Check if tensor is empty
bool Tensor<float>::empty() const { return this->data_.empty(); }

// Get tensor shape as vector
std::vector<uint32_t> Tensor<float>::shapes() const {
  CHECK(!this->data_.empty());
  return std::vector<uint32_t>{this->channels(), this->rows(), this->cols()};
}

// Get sub-shape (channels, rows, cols)
const std::vector<uint32_t> Tensor<float>::sub_shape() const {
  CHECK(!this->data_.empty());
  return std::vector<uint32_t>{this->channels(), this->rows(), this->cols()};
}

void Tensor<float>::set_data(const arma::fcube &data) {
  CHECK(data.n_rows == this->data_.n_rows)
      << data.n_rows << " != " << this->data_.n_rows;
  CHECK(data.n_cols == this->data_.n_cols)
      << data.n_cols << " != " << this->data_.n_cols;
  CHECK(data.n_slices == this->data_.n_slices)
      << data.n_slices << " != " << this->data_.n_slices;
  this->data_ = data;
}

float Tensor<float>::index(uint32_t position) const {
  CHECK(position < this->data_.size()) << "Tensor index Out of Bound";
  return this->data_.at(position);
}

float &Tensor<float>::index(uint32_t position) {
  CHECK(position < this->data_.size()) << "Tensor index Out of Bound";
  return this->data_.at(position);
}

float Tensor<float>::posi(uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  return this->data_.at(row, col, 0);
}

float &Tensor<float>::posi(uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  return this->data_.at(row, col, 0);
}

float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

float &Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

arma::fcube &Tensor<float>::data() { return this->data_; }

const arma::fcube &Tensor<float>::data() const { return this->data_; }

arma::fmat &Tensor<float>::slice(uint32_t channel) {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

const arma::fmat &Tensor<float>::slice(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

void Tensor<float>::Padding(const std::vector<uint32_t> &pads,
                            float padding_value) {
  CHECK(!this->data_.empty());
  CHECK_EQ(pads.size(), 4);

  const uint32_t pad_rows1 = pads.at(0);
  const uint32_t pad_rows2 = pads.at(1);
  const uint32_t pad_cols1 = pads.at(2);
  const uint32_t pad_cols2 = pads.at(3);

  const uint32_t channels = this->channels();
  const uint32_t rows = this->rows();
  const uint32_t cols = this->cols();

  const uint32_t new_rows = rows + pad_rows1 + pad_rows2;
  const uint32_t new_cols = cols + pad_cols1 + pad_cols2;

  if (pad_rows1 == 0 && pad_rows2 == 0 && pad_cols1 == 0 && pad_cols2 == 0) {
    return;
  }
  arma::fcube new_data;

  if (padding_value == 0.f) {
    new_data = arma::fcube(new_rows, new_cols, channels, arma::fill::zeros);
  } else {
    new_data = arma::fcube(new_rows, new_cols, channels, arma::fill::none);
    new_data.fill(padding_value);
  }

  new_data.subcube(pad_rows1, pad_cols1, 0, pad_rows1 + rows - 1,
                   pad_cols1 + cols - 1, channels - 1) = this->data_;

  this->data_ = std::move(new_data);

  if (this->raw_shape_.size() == 3) {
    this->raw_shape_ = {channels, new_rows, new_cols};
  } else if (this->raw_shape_.size() == 2) {
    this->raw_shape_ = {new_rows, new_cols};
  } else {
    this->raw_shape_ = {new_cols};
  }
}

void Tensor<float>::Fill(float value) {
  CHECK(!this->data_.empty());
  this->data_.fill(value);
}

void Tensor<float>::Fill(const std::vector<float> &values, bool row_major) {
  CHECK(!this->data_.empty());
  const uint32_t total_elems = this->data_.size();
  CHECK_EQ(values.size(), total_elems);
  if (row_major) {

    const uint32_t rows = this->rows();
    const uint32_t cols = this->cols();
    const uint32_t channels = this->data_.n_slices;
    const uint32_t planes = rows * cols;

    for (uint32_t it = 0; it < channels; it++) {
      auto &channel_data = this->data_.slice(it);

      const arma::fmat channel_wrapper(
          const_cast<float *>(values.data() + it * planes), this->cols(),
          this->rows(), false, true);
      channel_data = channel_wrapper.t();
    }
  } else {
    std::copy(values.begin(), values.end(), this->data_.memptr());
  }
}

void Tensor<float>::One() {
  CHECK(!this->data_.empty());
  this->Fill(1.f);
}

void Tensor<float>::Rand() {
  CHECK(!this->data_.empty());
  this->data_.randn();
}

std::vector<float> Tensor<float>::values(bool row_major) {
  CHECK_EQ(this->data_.empty(), false);
  std::vector<float> values(this->data_.size());

  if (!row_major) {
    std::copy(this->data_.mem, this->data_.mem + this->data_.size(),
              values.begin());
  } else {
    uint32_t index = 0;
    for (uint32_t it = 0; it < this->data_.n_slices; ++it) {
      const arma::fmat &channel = this->data_.slice(it).t();
      std::copy(channel.begin(), channel.end(), values.begin() + index);
      index += channel.size();
    }
    CHECK_EQ(index, values.size());
  }
  return values;
}

void Tensor<float>::Show() {
  for (uint32_t it = 0; it < this->channels(); it++) {
    LOG(INFO) << "Channel: " << it;
    LOG(INFO) << "\n" << this->data_.slice(it);
  }
}

void Tensor<float>::Reshape(const std::vector<uint32_t> &shapes,
                            bool row_major) {
  CHECK(!this->data_.empty());
  CHECK(!shapes.empty());
  const uint32_t origin_size = this->size();
  const uint32_t current_size =
      std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
  CHECK(shapes.size() <= 3);
  CHECK(current_size == origin_size);

  std::vector<float> values;
  if (row_major) {
    values = this->values(true);
  }

  if (shapes.size() == 3) {
    this->data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
    this->raw_shape_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
  } else if (shapes.size() == 2) {
    this->data_.reshape(shapes.at(0), shapes.at(1), 1);
    this->raw_shape_ = {shapes.at(0), shapes.at(1)};
  } else {
    this->data_.reshape(1, shapes.at(0), 1);
    this->raw_shape_ = {shapes.at(0)};
  }

  if (row_major) {
    this->Fill(values, true);
  }
}

void Tensor<float>::Flatten(bool row_major) {
  CHECK(!this->data_.empty());

  const uint32_t rows = this->rows();
  const uint32_t cols = this->cols();
  const uint32_t channels = this->channels();
  const uint32_t total_size = this->size();
  if (rows == 1 && channels == 1) {
    return;
  }
  arma::fcube flattened_data(1, total_size, 1);

  if (row_major) {
    const uint32_t tnums = rows * cols;

    for (uint32_t it = 0; it < channels; ++it) {
      arma::fmat transposed = this->data_.slice(it).t();
      float *target_ptr = flattened_data.memptr() + it * tnums;
      std::memcpy(target_ptr, transposed.memptr(), tnums * sizeof(float));
    }
  } else {
    std::memcpy(flattened_data.memptr(), this->data_.memptr(),
                total_size * sizeof(float));
  }
  this->data_ = std::move(flattened_data);
  this->raw_shape_ = {total_size};
}

void Tensor<float>::Transform(const std::function<float(float)> &filter) {
  CHECK(!this->data_.empty());
  this->data_.transform(filter);
}

float *Tensor<float>::raw_ptr() {
  CHECK(!this->data_.empty());
  return this->data_.memptr();
}

float *Tensor<float>::raw_ptr(uint32_t offset) {
  const uint32_t size = this->size();
  CHECK(!this->data_.empty());
  CHECK_LT(offset, size);
  return this->data_.memptr() + offset;
}

float *Tensor<float>::matrix_raw_ptr(uint32_t index) {
  CHECK_LT(index, this->channels());
  uint32_t offset = index * this->rows() * this->cols();
  CHECK_LE(offset, this->size());
  float *mem_ptr = this->raw_ptr() + offset;
  return mem_ptr;
}

float *Tensor<float>::tensor_raw_ptr(uint32_t index) { return this->raw_ptr(); }

<<<<<<< HEAD
}
=======
} // namespace dlc_inf
>>>>>>> 915227a76b08189ef98b37b621dcce6ee9d272b4
