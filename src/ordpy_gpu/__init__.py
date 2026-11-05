"""ordpy-gpu: GPU-Accelerated 2D Permutation Entropy, Statistical Complexity and Fisher Information in CUDA.

Motor de alto desempenho baseado em Códigos de Lehmer (argsort-free), Loteamento 3D e
Redução Entrópica por LUT vetorizada no dispositivo.
"""

from __future__ import annotations

from .core import (
    complexity_entropy,
    complexity_entropy_fisher,
    fisher_shannon,
    complexity_entropy_batch,
    complexity_entropy_fisher_batch,
    fisher_shannon_batch,
    ordinal_distribution_gpu,
    is_cuda_available,
    get_device_info,
)

__version__ = "1.1.0"
__author__ = "Fabio Linhares, Bruno Costa Nogueira, Rian Gabriel Santos Pinheiro, Fabiane da Silva Queiroz"
__all__ = [
    "complexity_entropy",
    "complexity_entropy_fisher",
    "fisher_shannon",
    "complexity_entropy_batch",
    "complexity_entropy_fisher_batch",
    "fisher_shannon_batch",
    "ordinal_distribution_gpu",
    "is_cuda_available",
    "get_device_info",
]
