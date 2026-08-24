"""Gerenciamento de Tabelas de Busca (Lookup Tables - LUT) para Redução Entrópica em GPU."""

from __future__ import annotations
import math
from typing import Dict, Tuple

try:
    import cupy as cp
except ImportError:
    cp = None

# Cache de LUTs por (M, n_states)
_LUT_CACHE: Dict[Tuple[int, int], Tuple[cp.ndarray, cp.ndarray, float]] = {}

def get_entropy_luts(M: int, n_states: int) -> Tuple[cp.ndarray, cp.ndarray, float]:
    """Retorna as tabelas pré-calculadas h_lut e m_lut e a constante de normalização q0.
    
    Parâmetros:
    -----------
    M : int
        Número total de janelas deslizantes na imagem.
    n_states : int
        Número de estados ordinais possíveis (k! = 24 para W=2, 362.880 para W=3).
    """
    if cp is None:
        raise RuntimeError("CuPy/CUDA indisponível.")
        
    key = (M, n_states)
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]
        
    c_arr = cp.arange(M + 1, dtype=cp.float64)
    p_arr = c_arr / float(M)
    
    # Shannon LUT: h_lut[c] = -(c/M) * ln(c/M) com convenção h_lut[0] = 0
    h_lut = cp.zeros(M + 1, dtype=cp.float64)
    nz = p_arr > 0
    h_lut[nz] = -p_arr[nz] * cp.log(p_arr[nz])
    
    # Jensen-Shannon LUT: m_lut[c] = -m * ln(m), onde m = 0.5 * (p + 1/n_states)
    u = 1.0 / n_states
    m_arr = 0.5 * (p_arr + u)
    m_lut = -m_arr * cp.log(m_arr)
    
    # Constante de Normalização Q0 (js_div_max = -0.5 * c1)
    if n_states == 24:
        c1 = (25.0 / 24.0) * math.log(25.0) - 2.0 * math.log(48.0) + math.log(24.0)
    else:
        n_p1 = float(n_states + 1)
        c1 = (n_p1 / n_states) * math.log(n_p1) - 2.0 * math.log(2.0 * n_states) + math.log(n_states)
    js_div_max = -0.5 * c1
    q0 = 1.0 / js_div_max
    
    _LUT_CACHE[key] = (h_lut, m_lut, q0)
    return h_lut, m_lut, q0
