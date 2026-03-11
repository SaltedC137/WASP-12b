/**
 * @file tensor.hpp
 * @author Aska Lyn
 * @brief A C++ implementation of multi-dimensional tensor data structures for deep learning
 * @details This header defines a templated Tensor class that supports 1D, 2D, and 3D tensor
 *          operations. The implementation uses Armadillo's fcube (float cube) as the underlying
 *          storage backend, providing efficient memory management and mathematical operations.
 * @date 2026-03-11 21:28:30
 */

#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <armadillo>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace dlc_inf
{
    /**
     * @brief Primary template declaration for Tensor class
     * @tparam T The data type stored in the tensor (default: float)
     * @details This is the primary template. Specializations are provided for
     *          uint8_t and float types.
     */
    template<typename T = float>
    class Tensor
    {
        // Primary template - implementation in specialized versions
    };

    /**
     * @brief Template specialization for uint8_t tensors
     * @details Optimized for 8-bit unsigned integer data, commonly used for
     *          quantized models or image data with pixel values [0, 255]
     */
    template<>
    class Tensor<uint8_t>
    {
        // Specialization for uint8_t - implementation pending
    };

    /**
     * @brief Template specialization for float tensors - the main implementation
     * @details This is the fully-featured tensor class for 32-bit floating-point
     *          data. It supports construction, copying, moving, element access,
     *          reshaping, and various data manipulation operations.
     *
     *          The tensor stores data in a 3D cube format (channels × rows × cols)
     *          using Armadillo's fcube. Memory layout follows row-major ordering
     *          for compatibility with standard C++ conventions.
     */
    template<>
    class Tensor<float>
    {
    public:
        /**
         * @brief Default constructor - creates an empty tensor
         * @details Initializes a tensor with no allocated data. The tensor will
         *          have zero dimensions until explicitly initialized.
         */
        explicit Tensor() = default;

        /**
         * @brief Construct a 3D tensor with specified dimensions
         * @param channels The number of channels (depth) in the tensor
         * @param rows The number of rows (height) in each channel
         * @param cols The number of columns (width) in each channel
         * @details Allocates memory for a 3D tensor with the given dimensions.
         *          All elements are initialized to zero by default.
         *          Example: Tensor(3, 224, 224) creates an RGB image tensor.
         */
        explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

        /**
         * @brief Construct a 1D tensor (vector) with specified size
         * @param size The number of elements in the 1D tensor
         * @details Creates a flat 1D tensor treated internally as (1, 1, size).
         *          Useful for bias vectors, flattened features, or 1D data.
         */
        explicit Tensor(uint32_t size);

        /**
         * @brief Construct a 2D tensor (matrix) with specified dimensions
         * @param rows The number of rows in the tensor
         * @param cols The number of columns in the tensor
         * @details Creates a 2D tensor treated internally as (1, rows, cols).
         *          Suitable for weight matrices or 2D feature maps.
         */
        explicit Tensor(uint32_t rows, uint32_t cols);

        /**
         * @brief Construct a tensor from a shape vector
         * @param shapes A vector specifying dimensions [channels, rows, cols] or [rows, cols] or [size]
         * @details Infers the tensor dimensionality from the number of elements
         *          in the shape vector: 1 element = 1D, 2 elements = 2D, 3 elements = 3D.
         */
        explicit Tensor(const std::vector<uint32_t> &shapes);

        /**
         * @brief Copy constructor - performs deep copy of tensor data
         * @param tensor The source tensor to copy from
         * @details Allocates new memory and copies all data from the source tensor.
         *          The copied tensor is completely independent of the original.
         */
        Tensor(const Tensor &tensor);

        /**
         * @brief Copy assignment operator - performs deep copy
         * @param tensor The source tensor to copy from
         * @return Reference to this tensor after assignment
         * @details Releases existing data and performs a deep copy from the source.
         *          Handles self-assignment safely.
         */
        Tensor<float> &operator=(const Tensor &tensor);

        /**
         * @brief Move constructor - transfers ownership from a temporary tensor
         * @param tensor The rvalue reference to the source tensor
         * @details Steals the internal data pointer and shape from the source tensor
         *          without copying. The source tensor is left in a valid but
         *          unspecified state. This is an O(1) operation.
         */
        Tensor(Tensor &&tensor) noexcept;

        /**
         * @brief Move assignment operator - transfers ownership from a temporary tensor
         * @param tensor The rvalue reference to the source tensor
         * @return Reference to this tensor after move assignment
         * @details Releases existing data and takes ownership of the source's data.
         *          More efficient than copy assignment for temporary objects.
         */
        Tensor<float> &operator=(Tensor &&tensor) noexcept;

        /**
         * @brief Get the number of rows in the tensor
         * @return uint32_t The row dimension size
         */
        uint32_t rows() const;

        /**
         * @brief Get the number of columns in the tensor
         * @return uint32_t The column dimension size
         */
        uint32_t cols() const;

        /**
         * @brief Get the number of channels in the tensor
         * @return uint32_t The channel (depth) dimension size
         */
        uint32_t channels() const;

        /**
         * @brief Get the total number of elements in the tensor
         * @return uint32_t The product of channels × rows × cols
         */
        uint32_t size() const;

        /**
         * @brief Check if the tensor contains no elements
         * @return true if size() == 0, false otherwise
         */
        bool empty() const;

        /**
         * @brief Get the shape of the tensor as a vector
         * @return std::vector<uint32_t> Shape in format [channels, rows, cols]
         */
        std::vector<uint32_t> shapes() const;

        /**
         * @brief Get the 2D sub-shape (rows and columns) excluding channels
         * @return std::vector<uint32_t> Shape in format [rows, cols]
         * @details Useful for operations that work on per-channel slices.
         */
        const std::vector<uint32_t> sub_shape() const;

        /**
         * @brief Set the tensor data from an Armadillo cube
         * @param data The fcube containing new data
         * @details Replaces the internal data with a copy of the provided cube.
         *          The tensor's shape is updated to match the input data.
         */
        void set_data(const arma::fcube &data);

        /**
         * @brief Access element using 1D linear index (const version)
         * @param position The linear index in row-major order
         * @return float The value at the specified position
         * @details Maps the 1D index to 3D coordinates internally.
         *          Bounds checking should be performed by the caller.
         */
        float index(uint32_t position) const;

        /**
         * @brief Access element using 1D linear index (mutable version)
         * @param position The linear index in row-major order
         * @return float& Reference to the value at the specified position
         * @details Allows modification of the element at the given index.
         */
        float &index(uint32_t position);

        /**
         * @brief Access element using 2D row-column indices (const version)
         * @param row The row index (0-based)
         * @param col The column index (0-based)
         * @return float The value at the specified position in channel 0
         * @details Primarily for 2D tensors (single channel). Accesses
         *          data at (channel=0, row, col).
         */
        float posi(uint32_t row, uint32_t col) const;

        /**
         * @brief Access element using 2D row-column indices (mutable version)
         * @param row The row index (0-based)
         * @param col The column index (0-based)
         * @return float& Reference to the value at the specified position
         */
        float &posi(uint32_t row, uint32_t col);

        /**
         * @brief Access element using 3D channel-row-column indices (const version)
         * @param channel The channel index (0-based)
         * @param row The row index (0-based)
         * @param col The column index (0-based)
         * @return float The value at the specified 3D position
         * @details Full 3D access for multi-channel tensors. This is the
         *          most general element access method.
         */
        float at(uint32_t channel, uint32_t row, uint32_t col) const;

        /**
         * @brief Access element using 3D channel-row-column indices (mutable version)
         * @param channel The channel index (0-based)
         * @param row The row index (0-based)
         * @param col The column index (0-based)
         * @return float& Reference to the value at the specified 3D position
         */
        float &at(uint32_t channel, uint32_t row, uint32_t col);

        /**
         * @brief Get mutable reference to the underlying Armadillo cube
         * @return arma::fcube& Reference to the internal data cube
         * @details Provides direct access to the raw data structure for
         *          advanced operations and interoperability with Armadillo.
         */
        arma::fcube &data();

        /**
         * @brief Get const reference to the underlying Armadillo cube
         * @return const arma::fcube& Const reference to the internal data cube
         * @details Read-only access to the raw data structure.
         */
        const arma::fcube &data() const;

        /**
         * @brief Get a mutable reference to a specific channel slice
         * @param channel The channel index to retrieve
         * @return arma::fmat& Reference to the 2D matrix at the specified channel
         * @details Returns the (rows × cols) matrix for the given channel.
         *          Useful for per-channel operations.
         */
        arma::fmat &slice(uint32_t channel);

        /**
         * @brief Get a const reference to a specific channel slice
         * @param channel The channel index to retrieve
         * @return const arma::fmat& Const reference to the 2D matrix
         * @details Read-only access to a channel's 2D data.
         */
        const arma::fmat &slice(uint32_t channel) const;

        /**
         * @brief Apply zero-padding to the tensor
         * @param sizes Padding sizes for each dimension [pad_top, pad_bottom, pad_left, pad_right]
         *              or similar format depending on implementation
         * @param padding_value The value to fill in padded regions (note: param name in
         *                      original code suggests it may default to 0)
         * @details Expands the tensor by adding padding around the borders.
         *          Commonly used in convolutional neural networks.
         */
        void Padding(const std::vector<uint32_t> &sizes, float padding_value);

        /**
         * @brief Fill the entire tensor with a single scalar value
         * @param value The value to fill
         * @details Sets all elements in the tensor to the specified value.
         */
        void Fill(float value);

        /**
         * @brief Fill the tensor with values from a vector
         * @param values The vector of values to copy into the tensor
         * @param row_major If true, values are interpreted in row-major order;
         *                  otherwise, column-major order
         * @details Copies elements from the input vector into the tensor.
         *          The vector size should match the tensor's total size.
         */
        void Fill(const std::vector<float> &values, bool row_major = true);

        /**
         * @brief Fill the tensor with ones
         * @details Sets all elements to 1.0f. Equivalent to Fill(1.0f).
         */
        void One();

        /**
         * @brief Fill the tensor with random values
         * @details Populates the tensor with random numbers. The specific
         *          distribution (uniform, normal, etc.) depends on implementation.
         */
        void Rand();

        /**
         * @brief Extract all tensor elements as a flat vector
         * @param row_major If true, elements are extracted in row-major order;
         *                  otherwise, column-major order
         * @return std::vector<float> A flat vector containing all tensor elements
         * @details The returned vector has size() elements. Useful for serialization
         *          or interfacing with other libraries.
         */
        std::vector<float> values(bool row_major = true);

        /**
         * @brief Display the tensor contents to standard output
         * @details Prints the tensor data in a human-readable format.
         *          Primarily for debugging purposes.
         */
        void Show();

        /**
         * @brief Reshape the tensor to new dimensions
         * @param shapes The new shape as a vector [channels, rows, cols] or subset
         * @param row_major If true, interprets data in row-major order during reshape
         * @details Changes the tensor's dimensions while preserving total element count.
         *          The underlying data is reinterpreted according to the new shape.
         * @note The new shape must be compatible (same total size) or reallocation occurs.
         */
        void Reshape(const std::vector<uint32_t> &shapes, bool row_major = true);

        /**
         * @brief Flatten the tensor to 1D
         * @param row_major If true, flattens in row-major order; otherwise, column-major
         * @details Converts the tensor to a 1D vector format (1, 1, size).
         *          Commonly used before fully connected layers.
         */
        void Flatten(bool row_major = true);

        /**
         * @brief Apply a transformation function to each element
         * @param filter A unary function that takes a float and returns a float
         * @details Applies the function element-wise, modifying the tensor in place.
         *          Example: Transform([](float x) { return std::max(0.0f, x); }) for ReLU.
         */
        void Transform(const std::function<float(float)> &filter);

        /**
         * @brief Get a raw pointer to the tensor's data buffer
         * @return float* Pointer to the first element
         * @details Provides direct memory access for interoperability with C APIs
         *          or performance-critical operations. Use with caution.
         */
        float *raw_ptr();

        /**
         * @brief Get a raw pointer with an offset
         * @param offset The element offset from the beginning
         * @return float* Pointer to the element at the offset
         * @details Useful for accessing subregions of the tensor data.
         */
        float *raw_ptr(uint32_t offset);

        /**
         * @brief Get a raw pointer to a specific matrix slice (channel)
         * @param index The channel index
         * @return float* Pointer to the first element of the specified channel
         * @details Returns a pointer to the beginning of the 2D slice at the given channel.
         */
        float *matrix_raw_ptr(uint32_t index);

        /**
         * @brief Get a raw pointer to a specific channel's data
         * @param index The channel index
         * @return float* Pointer to the data of the specified channel
         * @details Similar to matrix_raw_ptr, provides direct access to channel data.
         */
        float *tensor_raw_ptr(uint32_t index);

    private:
        std::vector<uint32_t> raw_shape_;  ///< Stores the original shape dimensions [channels, rows, cols]
        arma::fcube data_;                 ///< The underlying 3D data storage using Armadillo cube
    };

    // Type aliases for convenience
    using ften = Tensor<float>;                                    ///< Float tensor - shorthand alias
    using sft = std::shared_ptr<Tensor<float>>;                    ///< Shared pointer to float tensor
}

#endif