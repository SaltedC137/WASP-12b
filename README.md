# WASP12-b DL4Cpp

A C++ deep learning library implementing tensor operations and neural network components from scratch.

## ⚠️ Status: Under Construction

This project is currently **under active development**. The API is unstable and subject to change. Features are being incrementally added and refined.

## Project Overview

DL4Cpp aims to provide a lightweight, educational implementation of deep learning primitives in modern C++. The core components include:

- **Tensor**: Multi-dimensional tensor data structures (1D/2D/3D) with efficient memory management
- **Operations**: Element-wise operations, reshaping, padding, transformations, and slicing
- **Utilities**: CHECK macros for logging and assertion (similar to Google glog)
- **Backend**: Built on top of Armadillo for optimized linear algebra

## Project Structure

```
DL4Cpp/
├── include/              # Header files
│   ├── tensor.hpp        # Tensor class definition
│   └── check.hpp         # CHECK macros for logging and assertions
├── src/                  # Implementation files
│   ├── tensor.cpp        # Tensor class implementation
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

- [x] Tensor construction (1D, 2D, 3D)
- [x] Copy/Move semantics
- [x] Element access (`index`, `posi`, `at`)
- [x] Shape manipulation (`Reshape`, `Flatten`)
- [x] Padding (`Padding`)
- [x] Fill operations (`Fill`, `One`, `Rand`)
- [x] Transformation (`Transform`)
- [x] Data export (`values`, `Show`)
- [x] CHECK macros (`CHECK`, `CHECK_EQ`, `CHECK_LT`, `CHECK_LE`, `CHECK_GE`)

## TODO

- [ ] Convolution operations
- [ ] Pooling layers
- [ ] Activation functions
- [ ] Fully connected layers
- [ ] Backpropagation support
- [ ] LOG macros

---

*Last updated: 2026-03-11*
