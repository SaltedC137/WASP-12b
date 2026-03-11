# WASP12-b DL4Cpp

A C++ deep learning library implementing tensor operations and neural network components from scratch.

## ⚠️ Status: Under Construction

This project is currently **under active development**. The API is unstable and subject to change. Features are being incrementally added and refined.

## Project Overview

DL4Cpp aims to provide a lightweight, educational implementation of deep learning primitives in modern C++. The core components include:

- **Tensor**: Multi-dimensional tensor data structures with efficient memory management
- **Operations**: Element-wise operations, reshaping, padding, and transformations
- **Backend**: Built on top of Armadillo for optimized linear algebra

## Project Structure

```
DL4Cpp/
├── include/          # Header files
│   ├── tensor.hpp    # Tensor class definition
│   └── check.hpp     # Utility headers
├── src/              # Implementation files
│   └── tensor.cpp    # Tensor class implementation
├── example/          # Example usage code
│   └── ex.cpp        # Demonstration examples
└── CMakeLists.txt    # Build configuration
```

## Building

This project uses CMake as its build system.

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Requirements

- C++17 or later
- Armadillo library
- CMake 3.10+


---

*Last updated: 2026-03-11*
