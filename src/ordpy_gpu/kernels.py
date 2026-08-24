"""Gerenciamento e compilação dos kernels CUDA C++ (W=2 e W=3) para ordpy-gpu."""

from __future__ import annotations
import sys
from typing import Any

try:
    import cupy as cp
except ImportError:
    cp = None

KERNELS_CUDA_SRC = r'''
extern "C" {

// -----------------------------------------------------------------------------------------
// Kernel W=2 (k=4, 24 bins) com Shared Memory Cooperativa e Loteamento 3D
// -----------------------------------------------------------------------------------------
__global__ void kernel_lehmer_w2_fused_3d(const unsigned char* img_tensor, 
                                          unsigned int* hist_tensor,
                                          int Nr, int Nc, int W) {
    __shared__ unsigned int local_hist[24];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (tid < 24) local_hist[tid] = 0;
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z; // Canal cromático / lote

    int out_w = Nc - W + 1;
    int out_h = Nr - W + 1;

    if (x < out_w && y < out_h) {
        size_t offset = (size_t)z * Nr * Nc;
        unsigned char v0 = img_tensor[offset + y * Nc + x];
        unsigned char v1 = img_tensor[offset + y * Nc + (x + 1)];
        unsigned char v2 = img_tensor[offset + (y + 1) * Nc + x];
        unsigned char v3 = img_tensor[offset + (y + 1) * Nc + (x + 1)];

        // Contagem de Inversões de Lehmer (argsort-free estável)
        int l0 = (v1 < v0) + (v2 < v0) + (v3 < v0);
        int l1 = (v2 < v1) + (v3 < v1);
        int l2 = (v3 < v2);

        int pi = l0 * 6 + l1 * 2 + l2;
        atomicAdd(&local_hist[pi], 1);
    }
    __syncthreads();

    if (tid < 24) {
        atomicAdd(&hist_tensor[z * 24 + tid], local_hist[tid]);
    }
}

// -----------------------------------------------------------------------------------------
// Kernel W=3 (k=9, 362.880 bins) com Atômicos Globais e Loteamento 3D
// -----------------------------------------------------------------------------------------
__global__ void kernel_lehmer_w3_fused_3d(const unsigned char* img_tensor, 
                                          unsigned int* hist_tensor,
                                          int Nr, int Nc, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    int out_w = Nc - W + 1;
    int out_h = Nr - W + 1;

    if (x < out_w && y < out_h) {
        size_t offset = (size_t)z * Nr * Nc;
        unsigned char v[9];
        #pragma unroll
        for (int dy = 0; dy < 3; dy++) {
            #pragma unroll
            for (int dx = 0; dx < 3; dx++) {
                v[dy * 3 + dx] = img_tensor[offset + (y + dy) * Nc + (x + dx)];
            }
        }

        int l[9] = {0};
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            #pragma unroll
            for (int j = i + 1; j < 9; j++) {
                if (v[j] < v[i]) l[i]++;
            }
        }

        int pi = l[0] * 40320 + l[1] * 5040 + l[2] * 720 + l[3] * 120 +
                 l[4] * 24 + l[5] * 6 + l[6] * 2 + l[7] * 1;

        size_t hist_offset = (size_t)z * 362880 + pi;
        atomicAdd(&hist_tensor[hist_offset], 1);
    }
}

}
'''

_COMPILED_MODULE: Any = None

def get_cuda_kernels():
    """Retorna os ponteiros para os kernels compilados, compilando sob demanda."""
    global _COMPILED_MODULE
    if cp is None:
        raise RuntimeError("CuPy/CUDA não está disponível no ambiente. Instale 'cupy-cuda12x'.")
        
    if _COMPILED_MODULE is None:
        _COMPILED_MODULE = cp.RawModule(code=KERNELS_CUDA_SRC)
        
    k_w2 = _COMPILED_MODULE.get_function("kernel_lehmer_w2_fused_3d")
    k_w3 = _COMPILED_MODULE.get_function("kernel_lehmer_w3_fused_3d")
    return k_w2, k_w3
