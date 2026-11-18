"""Módulo central da API ordpy-gpu com interface 100% compatível com ordpy.

Suporta a trinca de quantificadores causais da Teoria da Informação:
- Entropia de Permutação Normalizada (H)
- Complexidade Estatística de Jensen-Shannon (C)
- Informação de Fisher Discreta (F)
"""

from __future__ import annotations
import math
from typing import Any, Tuple, Union, Optional
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from .kernels import get_cuda_kernels
from .lut import get_entropy_luts

# Vetor de mapeamento canônico de 24 estados: índice fatorádico de Lehmer (sigma^-1)
# para ordem lexicográfica ordinal estrita do ordpy (sigma).
# Garante identidade analítica exata (Diff < 10^-15) na Informação de Fisher (F).
LEHMER_TO_ORDPY_W2 = (
    0, 1, 2, 4, 3, 5, 6, 7, 12, 18, 13, 19,
    8, 10, 14, 20, 16, 22, 9, 11, 15, 21, 17, 23
)

def is_cuda_available() -> bool:
    """Verifica se há dispositivo CUDA e CuPy funcional no sistema."""
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False

def get_device_info() -> dict[str, Any]:
    """Retorna informações estruturadas da GPU ativa."""
    if not is_cuda_available():
        return {"status": "CUDA Indisponível"}
    props = cp.cuda.runtime.getDeviceProperties(0)
    vram_bytes = cp.cuda.Device(0).mem_info[1]
    return {
        "name": props["name"].decode("utf-8"),
        "vram_gb": vram_bytes / (1024**3),
        "sm_count": props["multiProcessorCount"],
        "compute_capability": f"{props['major']}.{props['minor']}",
        "cupy_version": cp.__version__,
    }

def _validate_window_dims(dx: int, dy: int) -> Tuple[int, int]:
    """Valida as dimensões de incorporação (dx, dy) e retorna (k, n_states)."""
    if dx < 1 or dy < 1:
        raise ValueError(f"Dimensões de janela devem ser >= 1 (dx={dx}, dy={dy}).")
    k = dx * dy
    if k < 2 or k > 9:
        raise ValueError(
            f"Ordem k=dx*dy={k} não suportada (dx={dx}, dy={dy}). "
            "Suporte ativo para 2 <= dx*dy <= 9 (ex.: 2x2, 2x3, 3x2, 1x4, 3x3)."
        )
    return k, math.factorial(k)


def complexity_entropy(
    data: np.ndarray,
    dx: int = 2,
    dy: int = 2,
    tau_x: int = 1,
    tau_y: int = 1,
    return_fisher: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, float]]:
    """Calcula a Entropia de Permutação 2D (H) e a Complexidade Estatística (C) de uma matriz 2D em GPU.
    
    Interface 100% compatível com ordpy.complexity_entropy(data, dx, dy).
    Se return_fisher=True, retorna a trinca (H, C, F).
    
    Parâmetros:
    -----------
    data : np.ndarray
        Matriz 2D de entrada (linhas x colunas).
    dx : int, default=2
        Largura da janela espacial (colunas, conforme convenção ordpy).
    dy : int, default=2
        Altura da janela espacial (linhas, conforme convenção ordpy).
    tau_x, tau_y : int, default=1
        Passo de amostragem espacial.
    return_fisher : bool, default=False
        Se True, calcula e retorna também a Informação de Fisher (F).
        
    Retorna:
    --------
    (H, C) ou (H, C, F) : Tuple[float, ...]
        H: Entropia de Permutação normalizada no intervalo [0, 1].
        C: Complexidade Estatística de Jensen-Shannon.
        F: Informação de Fisher discreta (se return_fisher=True).
    """
    _validate_window_dims(dx, dy)

    data_arr = np.asarray(data)
    if data_arr.ndim != 2:
        raise ValueError(f"Esperada matriz 2D, obtido array com {data_arr.ndim} dimensões.")

    stack_3d = data_arr[np.newaxis, :, :]  # Formato (1, Nr, Nc)
    if return_fisher:
        H_arr, C_arr, F_arr = complexity_entropy_batch(stack_3d, dx=dx, dy=dy, return_fisher=True)
        return float(H_arr[0]), float(C_arr[0]), float(F_arr[0])
    else:
        H_arr, C_arr = complexity_entropy_batch(stack_3d, dx=dx, dy=dy, return_fisher=False)
        return float(H_arr[0]), float(C_arr[0])

def complexity_entropy_fisher(
    data: np.ndarray,
    dx: int = 2,
    dy: int = 2,
    tau_x: int = 1,
    tau_y: int = 1,
) -> Tuple[float, float, float]:
    """Calcula a trinca completa (H, C, F) em GPU para uma matriz 2D."""
    res = complexity_entropy(data, dx=dx, dy=dy, tau_x=tau_x, tau_y=tau_y, return_fisher=True)
    return res # type: ignore

def fisher_shannon(
    data: np.ndarray,
    dx: int = 2,
    dy: int = 2,
    tau_x: int = 1,
    tau_y: int = 1,
) -> Tuple[float, float]:
    """Calcula a Entropia de Permutação (H) e a Informação de Fisher (F) em GPU.
    
    Interface 100% compatível com ordpy.fisher_shannon(data, dx, dy).
    """
    h, _, f = complexity_entropy_fisher(data, dx=dx, dy=dy, tau_x=tau_x, tau_y=tau_y)
    return h, f

def _launch_histogram_kernel(
    stack_gpu: "cp.ndarray",
    dx: int,
    dy: int,
    k: int,
    n_states: int,
) -> "cp.ndarray":
    """Despacha o kernel CUDA adequado (W=2, W=3 ou retangular Dx x Dy) e retorna hist_tensor."""
    B, Nr, Nc = stack_gpu.shape
    out_w = Nc - dx + 1
    out_h = Nr - dy + 1
    if out_w <= 0 or out_h <= 0:
        raise ValueError(
            f"Dimensões da matriz ({Nr}x{Nc}) são menores que a janela ({dy}x{dx})."
        )

    k_w2, k_w3, k_rect = get_cuda_kernels()
    block = (16, 16, 1)
    grid = ((out_w + 15) // 16, (out_h + 15) // 16, B)
    hist_tensor = cp.zeros((B, n_states), dtype=cp.uint32)

    if dx == 2 and dy == 2:
        k_w2(grid, block, (stack_gpu, hist_tensor, Nr, Nc, 2))
    elif dx == 3 and dy == 3:
        k_w3(grid, block, (stack_gpu, hist_tensor, Nr, Nc, 3))
    else:
        k_rect(grid, block, (stack_gpu, hist_tensor, Nr, Nc, dx, dy, k, n_states))
    return hist_tensor


def complexity_entropy_batch(
    stack: Union[np.ndarray, cp.ndarray],
    dx: int = 2,
    dy: int = 2,
    return_fisher: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calcula quantificadores para um tensor 3D (B, Nr, Nc) via Loteamento 3D CUDA.
    
    Parâmetros:
    -----------
    stack : np.ndarray ou cp.ndarray
        Tensor 3D de formato (B canais/imagens, Nr linhas, Nc colunas).
    dx, dy : int, default=2
        Dimensão da subgrade espacial (2 <= dx*dy <= 9).
    return_fisher : bool, default=False
        Se True, retorna a tupla de três vetores (H_vec, C_vec, F_vec).
        
    Retorna:
    --------
    (H_vec, C_vec) ou (H_vec, C_vec, F_vec) : Tuple[np.ndarray, ...]
    """
    if not is_cuda_available():
        raise RuntimeError("GPU CUDA/CuPy indisponível para execução acelerada.")

    k, n_states = _validate_window_dims(dx, dy)

    # Garante array contíguo em GPU tipo uint8
    if isinstance(stack, np.ndarray):
        if stack.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, Nr, Nc), obtido array com {stack.ndim} dimensões.")
        stack_gpu = cp.asarray(stack, dtype=cp.uint8)
    else:
        stack_gpu = stack
        if stack_gpu.ndim != 3:
            raise ValueError(f"Esperado tensor 3D (B, Nr, Nc), obtido cp.ndarray com {stack_gpu.ndim} dimensões.")

    B, Nr, Nc = stack_gpu.shape
    M = (Nr - dy + 1) * (Nc - dx + 1)
    if M <= 0 or (Nr - dy + 1) <= 0 or (Nc - dx + 1) <= 0:
        raise ValueError(f"Dimensões da matriz ({Nr}x{Nc}) são menores que o tamanho da janela ({dy}x{dx}).")

    h_lut, m_lut, q0 = get_entropy_luts(M, n_states)
    hist_tensor = _launch_histogram_kernel(stack_gpu, dx, dy, k, n_states)
        
    # Redução Entrópica por LUT vetorizada em VRAM
    h_vals = h_lut[hist_tensor]
    m_vals = m_lut[hist_tensor]
    
    H_raw = cp.sum(h_vals, axis=1)
    S_m = cp.sum(m_vals, axis=1)
    
    S_u = math.log(n_states)
    JS = S_m - 0.5 * H_raw - 0.5 * S_u
    
    H_norm = H_raw / S_u
    C_stat = q0 * JS * H_norm
    
    if return_fisher:
        # Cômputo da Informação de Fisher (F) com paridade IEEE 754 rigorosa frente ao ordpy
        if k == 4:
            map_gpu = cp.asarray(LEHMER_TO_ORDPY_W2, dtype=cp.int32)
            hist_ordpy = cp.zeros_like(hist_tensor, dtype=cp.float64)
            hist_ordpy[:, map_gpu] = hist_tensor.astype(cp.float64)
            P = hist_ordpy / float(M)
        else:
            P = hist_tensor.astype(cp.float64) / float(M)
            
        sqrt_P = cp.sqrt(P)
        diff_sqrt = cp.diff(sqrt_P, axis=1)
        F_stat = 0.5 * cp.sum(diff_sqrt ** 2, axis=1)
        
        # Paridade com convenção ordpy: distribuição degenerada mono-estado resulta em F=1.0
        single_state_mask = (P == 1.0).any(axis=1)
        F_stat = cp.where(single_state_mask, 1.0, F_stat)
        
        return H_norm.get(), C_stat.get(), F_stat.get()
        
    return H_norm.get(), C_stat.get()

def complexity_entropy_fisher_batch(
    stack: Union[np.ndarray, cp.ndarray],
    dx: int = 2,
    dy: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula a trinca (H, C, F) em lote tridimensional na GPU."""
    res = complexity_entropy_batch(stack, dx=dx, dy=dy, return_fisher=True)
    return res # type: ignore

def fisher_shannon_batch(
    stack: Union[np.ndarray, cp.ndarray],
    dx: int = 2,
    dy: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula (H, F) em lote tridimensional na GPU."""
    h_vec, _, f_vec = complexity_entropy_fisher_batch(stack, dx=dx, dy=dy)
    return h_vec, f_vec

def ordinal_distribution_gpu(
    stack: Union[np.ndarray, cp.ndarray],
    dx: int = 2,
    dy: int = 2,
) -> np.ndarray:
    """Retorna os histogramas ordinais de frequências absolutas (B x k!) computados na GPU."""
    if not is_cuda_available():
        raise RuntimeError("GPU CUDA indisponível.")
    k, n_states = _validate_window_dims(dx, dy)
    if isinstance(stack, np.ndarray):
        if stack.ndim == 2:
            stack = stack[np.newaxis, :, :]
        stack_gpu = cp.asarray(stack, dtype=cp.uint8)
    else:
        stack_gpu = stack if stack.ndim == 3 else stack[cp.newaxis, :, :]

    hist_tensor = _launch_histogram_kernel(stack_gpu, dx, dy, k, n_states)
    return hist_tensor.get()
