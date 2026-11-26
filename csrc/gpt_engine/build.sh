#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
NVCC="$CUDA_HOME/bin/nvcc"
HOST_CXX="${HOST_CXX:-/usr/bin/g++-15}"

echo "Building libindextts_gpt.so (sm_89 + sm_120 + compute_89 PTX)..."
"$NVCC" -O3 -std=c++17 \
    -ccbin "$HOST_CXX" \
    -Xcompiler -fPIC \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_89,code=sm_120 \
    -gencode arch=compute_89,code=compute_89 \
    --expt-relaxed-constexpr \
    -shared \
    -o "$DIR/libindextts_gpt.so" \
    "$DIR/gpt_engine.cu" \
    -lcublas

echo "Verifying embedded ELF/PTX architectures:"
"$CUDA_HOME/bin/cuobjdump" --list-elf "$DIR/libindextts_gpt.so"
"$CUDA_HOME/bin/cuobjdump" --list-ptx "$DIR/libindextts_gpt.so"
ls -lh "$DIR/libindextts_gpt.so"
