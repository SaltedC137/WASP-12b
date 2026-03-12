# WASP-12b DL4Cpp

A C++ deep learning library implementing tensor operations and neural network components from scratch.

## ⚠️ Status: Under Construction

This project is currently **under active development**. The API is unstable and subject to change. Features are being incrementally added and refined.

## Project Overview

DL4Cpp aims to provide a lightweight, educational implementation of deep learning primitives in modern C++. The core components include:

- **Tensor**: Multi-dimensional tensor data structures (1D/2D/3D) with efficient memory management
- **Math Operations**: Element-wise arithmetic, matrix multiplication, scalar operations, and transformations
- **Utilities**: CHECK macros for logging and assertion (similar to Google glog)
- **Backend**: Built on top of Armadillo for optimized linear algebra

## Project Structure

```
DL4Cpp/
├── include/              # Header files
│   ├── tensor.hpp        # Tensor class definition
│   ├── tensor_math.hpp   # Tensor mathematical operations
│   ├── tensor_nn.hpp     # Neural network components (TODO)
│   ├── check.hpp         # CHECK macros for logging and assertions
│   └── log.hpp           # Logging utilities
├── src/                  # Implementation files
│   ├── tensor.cpp        # Tensor class implementation
│   ├── tensor_math.cpp   # Tensor math operations implementation (TODO)
│   └── check.cpp         # FMessageLogger and FMessageVoidify implementation
├── example/              # Example usage code
│   └── ex.cpp            # Demonstration examples
└── CMakeLists.txt        # Build configuration
```

## Building

This project uses CMake as its build system.

```bash
mkdir build && cd build

cmake ..

cmake --build .

cmake --install . --prefix .
```

### Build Output

- **Static Library**: `build/DL4Cpp/libDL4Cpp.a`
- **Test Executable**: `build/DL4Cpp/dl4cpp_test.exe`

## Requirements

- C++20 or later
- Armadillo library
- CMake 3.14+

## Implemented Features

### Core Tensor Operations
- [x] Tensor construction (1D, 2D, 3D)
- [x] Copy/Move semantics
- [x] Element access (`index`, `posi`, `at`)
- [x] Shape manipulation (`Reshape`, `Flatten`)
- [x] Padding (`Padding`)
- [x] Fill operations (`Fill`, `One`, `Rand`)
- [x] Transformation (`Transform`)
- [x] Data export (`values`, `Show`)
- [x] Raw pointer access (`raw_ptr`, `matrix_raw_ptr`)

### Math Operations (Header Only)
- [x] Element-wise operations (`ElementAdd`, `ElementSub`, `ElementMultiply`, `ElementDivide`)
- [x] Matrix multiplication (`Matmul`)
- [x] Scalar operations (`AddScalar`, `SubScalar`, `MultiplyScalar`, `DivideScalar`)
- [x] Element-wise exponential (`ElementExp`)
- [x] Element-wise clipping (`ElementClip`)
- [x] Functional interface (inline functions returning `std::shared_ptr`)

### Utilities
- [x] CHECK macros (`CHECK`, `CHECK_EQ`, `CHECK_LT`, `CHECK_LE`, `CHECK_GE`)

## TODO

### High Priority
- [ ] Implement math operations in `src/tensor_math.cpp`
- [ ] Convolution operations (Conv2D, Conv3D)
- [ ] Pooling layers (MaxPool, AvgPool)
- [ ] Activation functions (ReLU, Sigmoid, Tanh, Softmax)
- [ ] Fully connected layers (Dense/Linear)
- [ ] Backpropagation support and gradient computation
- [ ] Loss functions (MSE, CrossEntropy)
- [ ] Optimizers (SGD, Adam)

### Medium Priority
- [ ] LOG macros implementation
- [ ] Batch normalization
- [ ] Dropout layer
- [ ] Data loading utilities
- [ ] Model serialization (save/load weights)
- [ ] Unit tests for all components

### Low Priority
- [ ] GPU acceleration support (CUDA backend)
- [ ] Multi-threading for parallel operations
- [ ] Documentation website (Doxygen)
- [ ] Python bindings (pybind11)
- [ ] Example tutorials and benchmarks

---

*Last updated: 2026-03-12*
