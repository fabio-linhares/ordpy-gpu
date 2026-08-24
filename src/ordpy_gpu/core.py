"""Módulo central da API ordpy-gpu com interface 100% compatível com ordpy."""

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

def complexity_entropy(
    data: np.ndarray,
    dx: int = 2,
    dy: int = 2,
    tau_x: int = 1,
    tau_y: int = 1,
) -> Tuple[float, float]:
    """Calcula a Entropia de Permutação 2D (H) e a Complexidade Estatística (C) de uma matriz 2D em GPU.
    
    Interface 100% compatível com ordpy.complexity_entropy(data, dx, dy).
    
    Parâmetros:
    -----------
    data : np.ndarray
        Matriz 2D de entrada (linhas x colunas).
    dx : int, default=2
        Largura da janela espacial (dx=2 ou dx=3).
    dy : int, default=2
        Altura da janela espacial (dy deve ser igual a dx).
    tau_x, tau_y : int, default=1
        Passo de amostragem espacial.
        
    Retorna:
    --------
    H : float
        Entropia de Permutação normalizada no intervalo [0, 1].
    C : float
        Complexidade Estatística de Jensen-Shannon.
    """
    if dx != dy:
        raise ValueError(f"Janelas não-quadradas não são suportadas atualmente (dx={dx}, dy={dy}).")
    W = dx
    if W not in (2, 3):
        raise ValueError(f"Ordem de janela W={W} não suportada. Suporte ativo para W=2 (24 estados) e W=3 (362.880 estados).")
        
    data_arr = np.asarray(data)
    if data_arr.ndim != 2:
        raise ValueError(f"Esperada matriz 2D, obtido array com {data_arr.ndim} dimensões.")
        
    stack_3d = data_arr[np.newaxis, :, :] # Formato (1, Nr, Nc)
    H_arr, C_arr = complexity_entropy_batch(stack_3d, dx=W, dy=W)
    return float(H_arr[0]), float(C_arr[0])

def complexity_entropy_batch(
    stack: Union[np.ndarray, cp.ndarray],
    dx: int = 2,
    dy: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula H e C para um tensor tridimensional (B, Nr, Nc) simultaneamente via Loteamento 3D CUDA.
    
    Parâmetros:
    -----------
    stack : np.ndarray ou cp.ndarray
        Tensor 3D de formato (B canais/imagens, Nr linhas, Nc colunas).
    dx, dy : int, default=2
        Dimensão da subgrade espacial (W=2 ou W=3).
        
    Retorna:
    --------
    H_vec : np.ndarray
        Vetor 1D de tamanho B contendo a Entropia de Permutação normalizada de cada canal.
    C_vec : np.ndarray
        Vetor 1D de tamanho B contendo a Complexidade Estatística de cada canal.
    """
    if not is_cuda_available():
        raise RuntimeError("GPU CUDA/CuPy indisponível para execução acelerada.")
        
    if dx != dy:
        raise ValueError(f"Janelas não-quadradas não suportadas (dx={dx}, dy={dy}).")
    W = dx
    if W not in (2, 3):
        raise ValueError(f"Ordem de janela W={W} não suportada. Use W=2 ou W=3.")
        
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
    M = (Nr - W + 1) * (Nc - W + 1)
    if M <= 0:
        raise ValueError(f"Dimensões da matriz ({Nr}x{Nc}) são menores que o tamanho da janela ({W}x{W}).")
        
    n_states = 24 if W == 2 else 362880
    k_w2, k_w3 = get_cuda_kernels()
    h_lut, m_lut, q0 = get_entropy_luts(M, n_states)
    
    block = (16, 16, 1)
    grid = ((Nc - W + 1 + 15) // 16, (Nr - W + 1 + 15) // 16, B)
    
    hist_tensor = cp.zeros((B, n_states), dtype=cp.uint32)
    if W == 2:
        k_w2(grid, block, (stack_gpu, hist_tensor, Nr, Nc, W))
    else:
        k_w3(grid, block, (stack_gpu, hist_tensor, Nr, Nc, W))
        
    # Redução Entrópica por LUT vetorizada em VRAM
    h_vals = h_lut[hist_tensor]
    m_vals = m_lut[hist_tensor]
    
    H_raw = cp.sum(h_vals, axis=1)
    S_m = cp.sum(m_vals, axis=1)
    
    S_u = math.log(n_states)
    JS = S_m - 0.5 * H_raw - 0.5 * S_u
    
    H_norm = H_raw / S_u
    C_stat = q0 * JS * H_norm
    
    return H_norm.get(), C_stat.get()

def ordinal_distribution_gpu(
    stack: Union[np.ndarray, cp.ndarray],
    dx: int = 2,
    dy: int = 2,
) -> np.ndarray:
    """Retorna os histogramas ordinais de frequências absolutas (B x k!) computados na GPU."""
    if not is_cuda_available():
        raise RuntimeError("GPU CUDA indisponível.")
    W = dx
    if isinstance(stack, np.ndarray):
        if stack.ndim == 2:
            stack = stack[np.newaxis, :, :]
        stack_gpu = cp.asarray(stack, dtype=cp.uint8)
    else:
        stack_gpu = stack if stack.ndim == 3 else stack[cp.newaxis, :, :]
        
    B, Nr, Nc = stack_gpu.shape
    n_states = 24 if W == 2 else 362880
    k_w2, k_w3 = get_cuda_kernels()
    
    block = (16, 16, 1)
    grid = ((Nc - W + 1 + 15) // 16, (Nr - W + 1 + 15) // 16, B)
    hist_tensor = cp.zeros((B, n_states), dtype=cp.uint32)
    
    if W == 2:
        k_w2(grid, block, (stack_gpu, hist_tensor, Nr, Nc, W))
    else:
        k_w3(grid, block, (stack_gpu, hist_tensor, Nr, Nc, W))
        
    return hist_tensor.get()
