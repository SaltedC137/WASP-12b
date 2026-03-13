# WASP-12b DL4Cpp

A C++ deep learning library implementing tensor operations and neural network components from scratch.

## ⚠️ Status: Under Construction

This project is currently **under active development**. The API is unstable and subject to change. Features are being incrementally added and refined.

## Project Overview

DL4Cpp aims to provide a lightweight, educational implementation of deep learning primitives in modern C++. The core components include:

- **Tensor**: Multi-dimensional tensor data structures (1D/2D/3D) with efficient memory management
- **Math Operations**: Element-wise arithmetic, matrix multiplication, scalar operations, and transformations
- **Utilities**: CHECK macros for logging and assertions (similar to Google glog)
- **Backend**: Built on top of Armadillo for optimized linear algebra

## Project Structure

```
WASP-12b/
├── DL4Cpp/
│   ├── include/              # Header files
│   │   ├── core/
│   │   │   ├── tensor.hpp        # Tensor class definition
│   │   │   ├── tensor_math.hpp   # Tensor mathematical operations
│   │   │   └── tensor_nn.hpp     # Neural network components (TODO)
│   │   ├── check.hpp         # CHECK macros for logging and assertions
│   │   └── log.hpp           # Logging utilities
│   ├── src/                  # Implementation files
│   │   ├── core/
│   │   │   ├── tensor.cpp        # Tensor class implementation
│   │   │   ├── tensor_math.cpp   # Tensor math operations implementation (TODO)
│   │   │   └── tensor_nn.cpp     # Neural network components implementation (TODO)
│   │   └── log.cpp         # Logging implementation
│   ├── example/              # Example code
│   │   ├── ex.cpp            # Feature demonstration examples
│   │   └── bench.cpp         # Performance benchmarks
│   └── CMakeLists.txt        # Build configuration
├── out/                      # Build output directory
│   └── bin/                  # Final output files (executables, libraries, headers)
├── build/                    # CMake build directory
└── CMakeLists.txt            # Root project configuration
```

## Building

This project uses CMake as its build system.

### Quick Start

```bash
# Configure and build
mkdir build && cd build
cmake ..
cmake --build .
```

### Full Build Process (with Installation)

```bash
# 1. Create build directory
mkdir build && cd build

# 2. Configure project
cmake ..

# 3. Build project
cmake --build .

# 4. Install to out/bin/
cmake --install . --prefix ../out/bin
```

### Using Presets (Recommended)

The project provides `CMakePresets.json` for convenient configuration:

```bash
# Configure using preset
cmake --preset=default

# Build using preset
cmake --build --preset=default
```

### Build Artifacts

After building, output files are located in the `out/bin/` directory:

| File | Description |
|------|-------------|
| `lib/DL4Cpp.lib` | Static library (Windows) |
| `lib/libDL4Cpp.a` | Static library (Linux/macOS) |
| `bin/dl4cpp_test.exe` | Test executable |
| `bin/dl4cpp_bench.exe` | Benchmark executable |
| `include/` | Header files copy |

### Running Tests

```bash
cd build
ctest --output-on-failure
```

Or run the test executable directly:

```bash
./out/bin/dl4cpp_test.exe
```

## Requirements

- **C++23** or later
- **Armadillo** linear algebra library
- **CMake 3.14+**

### Installing Dependencies

Using vcpkg (recommended):

```bash
# Set VCPKG_ROOT environment variable
set VCPKG_ROOT=C:\path\to\vcpkg  # Windows

# Install Armadillo
vcpkg install armadillo
```

Or use system package managers:

```bash
# Ubuntu/Debian
sudo apt-get install libarmadillo-dev

# macOS (Homebrew)
brew install armadillo

# Windows (vcpkg)
vcpkg install armadillo:x64-windows
```

## Implemented Features

### Core Tensor Operations
- [x] Tensor construction (1D, 2D, 3D)
- [x] Copy/Move semantics
- [x] Element access (`index`, `posi`, `at`)
- [x] Shape manipulation (`Reshape`, `Flatten`)
- [x] Padding operations (`Padding`)
- [x] Fill operations (`Fill`, `One`, `Rand`)
- [x] Transformation operations (`Transform`)
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
- [x] Logging (`FMessageLogger`, `FMessageVoidify`)

## TODO

### High Priority
- [ ] Implement math operations in `src/core/tensor_math.cpp`
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

## License

This project is licensed under the MIT License.

---

*Last updated: 2026-03-13*
