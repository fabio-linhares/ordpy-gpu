#!/usr/bin/env python3
"""Exemplo de inicialização rápida (Quickstart) com ordpy-gpu."""

import time
import numpy as np
import ordpy_gpu

def main():
    print("=" * 70)
    print(" ORDPY-GPU: EXEMPLO DE DEMONSTRAÇÃO RÁPIDA")
    print("=" * 70)
    
    # 1. Informações de Hardware
    dev = ordpy_gpu.get_device_info()
    print(f"[*] Dispositivo Ativo : {dev.get('name', 'N/A')}")
    print(f"[*] VRAM Total        : {dev.get('vram_gb', 0):.2f} GB")
    print(f"[*] Multiprocessors   : {dev.get('sm_count', 0)} SMs")
    print("-" * 70)
    
    # 2. Processando lote de 13 canais cromáticos de 800x800
    B, H, W = 13, 800, 800
    print(f"[*] Gerando lote de entrada: {B} canais x {H}x{W} pixels ({B * (H-1) * (W-1):,} janelas)...")
    stack = np.random.randint(0, 256, size=(B, H, W), dtype=np.uint8)
    
    # 3. Execução Acelerada em GPU
    t0 = time.perf_counter()
    H_vec, C_vec = ordpy_gpu.complexity_entropy_batch(stack, dx=2, dy=2)
    t1 = time.perf_counter()
    
    tempo_ms = (t1 - t0) * 1000.0
    print(f"[✓] Lote de 13 canais processado com sucesso em: {tempo_ms:.3f} ms!")
    print(f"    - Entropia Média (H)    : {np.mean(H_vec):.6f}")
    print(f"    - Complexidade Média (C): {np.mean(C_vec):.6f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
