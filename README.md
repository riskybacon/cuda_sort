# Introduction

CUDA implementation of Radix sort on unsigned ints. The goal was to get a
version of radix sort working, not to make a fast version of radix sort.

# Building and Running

Dependencies:

1. A CUDA compatible host compiler
1. CUDA
2. CMake

```
git clone git@github.com:riskybacon/cuda_blelloch_scan.git
cd cuda_blelloch_scan
mkdir build
cd build
cmake ..
make
./sort 4000000
```
