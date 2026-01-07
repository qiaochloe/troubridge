#!/bin/bash
set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Install Bazel
if ! command -v bazel &> /dev/null; then
    echo "Installing Bazel..."
    cd ~
    wget https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64
    chmod +x bazelisk-linux-amd64
    ./bazelisk-linux-amd64
    sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
    cd "$SCRIPT_DIR"
fi

# Set up tokenizers-cpp
if [ ! -d "third_party/tokenizers-cpp" ]; then
    echo "Setting up tokenizers-cpp..."
    git submodule add https://github.com/mlc-ai/tokenizers-cpp.git third_party/tokenizers-cpp || true
    git submodule update --init --recursive
fi

# Install cmake and Rust if needed
if ! command -v cmake &> /dev/null; then
    echo "Installing cmake..."
    sudo apt update && sudo apt install -y cmake
fi

if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
fi

# Build tokenizers-cpp
if [ ! -f "third_party/tokenizers-cpp/build/libtokenizers_cpp.a" ]; then
    echo "Building tokenizers-cpp..."
    cd third_party/tokenizers-cpp
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cd "$SCRIPT_DIR"
fi

# Copy BUILD file for tokenizers-cpp
if [ -f "setup/TOKENIZERS_BUILD" ]; then
    cp setup/TOKENIZERS_BUILD third_party/tokenizers-cpp/BUILD
fi

# Set up libtorch
if [ ! -d "third_party/libtorch" ]; then
    echo "Setting up libtorch..."
    cd third_party
    wget -q https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.9.1%2Bcpu.zip
    unzip -q libtorch-shared-with-deps-2.9.1+cpu.zip
    rm libtorch-shared-with-deps-2.9.1+cpu.zip
    cd "$SCRIPT_DIR"
fi

# Copy BUILD file for libtorch
if [ -f "setup/LIBTORCH_BUILD" ]; then
    cp setup/LIBTORCH_BUILD third_party/libtorch/BUILD
fi

# Set up LD_LIBRARY_PATH permanently
LIBTORCH_LIB_PATH="$SCRIPT_DIR/third_party/libtorch/lib"
if [ -d "$LIBTORCH_LIB_PATH" ]; then
    # Determine which shell config file to use
    if [ -f "$HOME/.bashrc" ]; then
        SHELL_CONFIG="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_CONFIG="$HOME/.bash_profile"
    else
        SHELL_CONFIG="$HOME/.bashrc"
        touch "$SHELL_CONFIG"
    fi

    # Check if already configured
    if ! grep -q "third_party/libtorch/lib" "$SHELL_CONFIG" 2>/dev/null; then
        echo "" >> "$SHELL_CONFIG"
        echo "# Added by setup.sh - libtorch library path" >> "$SHELL_CONFIG"
        echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$LIBTORCH_LIB_PATH" >> "$SHELL_CONFIG"
        echo "Added libtorch library path to $SHELL_CONFIG"
    fi

    # Set for current session
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBTORCH_LIB_PATH
    echo "LD_LIBRARY_PATH set for current session"
fi

# Build and test
echo "Building hello_main..."
bazel build //tcmalloc/testing:hello_main
echo "Running hello_main..."
bazel run //tcmalloc/testing:hello_main
