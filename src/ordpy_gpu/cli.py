"""Interface de Linha de Comando (CLI) e Sistema de Ajuda Interativa para ordpy-gpu.

Permite ao usuário consultar informações de hardware, obter ajuda contextual no
estilo de comandos Linux/man e auditar a veracidade matemática do motor GPU frente
ao pipeline canônico de referência da CPU.
"""

from __future__ import annotations
import sys
import time
import argparse
import numpy as np

try:
    import ordpy
except ImportError:
    ordpy = None

import ordpy_gpu
from .core import is_cuda_available, get_device_info, complexity_entropy_batch

# Códigos de cores ANSI para formatação no terminal Linux
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"

def print_banner():
    banner = f"""{CYAN}{BOLD}
   ____           __               ____ _____  __  __
  / __ \_________  / /_  __  __     / __ )/ __ \/ / / /
 / / / / ___/ __ \/ __ \/ / / /___ / /_/ / /_/ / / / / 
/ /_/ / /  / /_/ / /_/ / /_/ /___// ____/ ____/ /_/ /  
\____/_/  / .___/_.___/\__, /    /_/   /_/    \____/   
         /_/          /____/                           
{RESET}{BOLD}Aceleração de Entropia de Permutação 2D e Complexidade Estatística em GPU{RESET}
{BLUE}Algoritmo argsort-free baseado em Códigos de Lehmer e Redução por LUT{RESET}
"""
    print(banner)

def cmd_info(args):
    """Exibe informações detalhadas do hardware e do ambiente CUDA."""
    print_banner()
    print(f"{BOLD}=== DIAGNÓSTICO DE HARDWARE E AMBIENTE ==={RESET}\n")
    
    if not is_cuda_available():
        print(f"{RED}[✗] GPU CUDA / CuPy NÃO disponível no ambiente.{RESET}")
        print("    Certifique-se de ter drivers NVIDIA e o pacote cupy-cuda12x instalados.\n")
        return 1
        
    info = get_device_info()
    ordpy_ver = getattr(ordpy, "__version__", "1.2.2") if ordpy is not None else "Não instalado (opcional)"
    
    print(f"  {GREEN}[✓] Dispositivo Ativo           :{RESET} {BOLD}{info.get('name')}{RESET}")
    print(f"  {GREEN}[✓] Memória Global de VRAM      :{RESET} {info.get('vram_gb', 0):.2f} GB")
    print(f"  {GREEN}[✓] Compute Capability          :{RESET} CUDA {info.get('compute_capability')}")
    print(f"  {GREEN}[✓] Streaming Multiprocessors   :{RESET} {info.get('sm_count')} SMs")
    print(f"  {GREEN}[✓] Versão do CuPy / NVRTC      :{RESET} CuPy {info.get('cupy_version')}")
    print(f"  {GREEN}[✓] Versão do ordpy (CPU Canônico):{RESET} ordpy {ordpy_ver}")
    print(f"  {GREEN}[✓] Versão do ordpy-gpu         :{RESET} {ordpy_gpu.__version__}\n")
    print(f"{CYAN}Tudo pronto para computação acelerada em sub-milissegundos!{RESET}\n")
    return 0

def cmd_verify(args):
    """Audita a veracidade das respostas do ordpy-gpu contra o ordpy CPU."""
    print_banner()
    print(f"{BOLD}=== PROVA DE VERACIDADE E EQUIVALÊNCIA NUMÉRICA (CPU vs GPU) ==={RESET}\n")
    
    if not is_cuda_available():
        print(f"{RED}[ERRO] GPU CUDA indisponível para verificação.{RESET}\n")
        return 1
        
    if ordpy is None:
        print(f"{RED}[ERRO] O pacote 'ordpy' (CPU) não está instalado no ambiente.{RESET}")
        print("    Para auditar e comparar com a CPU de referência, instale: pip install ordpy\n")
        return 1
        
    res = args.size
    channels = args.channels
    W = args.order
    
    print(f"[*] Gerando lote de teste: {channels} canais x {res}x{res} pixels (W={W})...")
    np.random.seed(args.seed)
    test_stack = np.random.randint(0, 256, size=(channels, res, res), dtype=np.uint8)
    
    print(f"[*] Executando na CPU ({BOLD}ordpy canônico sequencial{RESET})...")
    t0_cpu = time.perf_counter()
    H_cpu = np.zeros(channels, dtype=np.float64)
    C_cpu = np.zeros(channels, dtype=np.float64)
    for ch in range(channels):
        h, c = ordpy.complexity_entropy(test_stack[ch], dx=W, dy=W)
        H_cpu[ch] = h
        C_cpu[ch] = c
    t_cpu = (time.perf_counter() - t0_cpu) * 1000.0
    
    print(f"[*] Executando na GPU ({BOLD}ordpy-gpu com Códigos de Lehmer e LUT{RESET})...")
    t0_gpu = time.perf_counter()
    H_gpu, C_gpu = complexity_entropy_batch(test_stack, dx=W, dy=W)
    t_gpu = (time.perf_counter() - t0_gpu) * 1000.0
    
    diff_H = np.abs(H_cpu - H_gpu)
    diff_C = np.abs(C_cpu - C_gpu)
    max_dh = np.max(diff_H)
    max_dc = np.max(diff_C)
    speedup = t_cpu / t_gpu if t_gpu > 0 else 0
    
    print("\n" + "=" * 95)
    print(f"{'Canal':^8} | {'H (CPU)':^14} | {'H (GPU)':^14} | {'Δ H (Resíduo)':^15} | {'Δ C (Resíduo)':^15} | {'Status':^12}")
    print("=" * 95)
    
    for ch in range(min(channels, 13)):
        status = f"{GREEN}APROVADO{RESET}" if (diff_H[ch] <= 1e-14 and diff_C[ch] <= 1e-14) else f"{RED}REPROVADO{RESET}"
        print(f"{ch:^8d} | {H_cpu[ch]:^14.8f} | {H_gpu[ch]:^14.8f} | {diff_H[ch]:^15.2e} | {diff_C[ch]:^15.2e} | {status:^12}")
        
    print("=" * 95)
    print(f"\n{BOLD}RELATÓRIO DE AUDITORIA DE PRECISÃO E VELOCIDADE:{RESET}")
    print(f"  • {BOLD}Maior Resíduo Absoluto em H :{RESET} {GREEN}{max_dh:.2e}{RESET} (Limite de máquina IEEE 754 float64)")
    print(f"  • {BOLD}Maior Resíduo Absoluto em C :{RESET} {GREEN}{max_dc:.2e}{RESET} (Limite de máquina IEEE 754 float64)")
    print(f"  • {BOLD}Tempo Total na CPU (ordpy)   :{RESET} {YELLOW}{t_cpu:.2f} ms{RESET} ({t_cpu/1000.0:.2f} segundos)")
    print(f"  • {BOLD}Tempo Total na GPU (ordpy-gpu):{RESET} {GREEN}{t_gpu:.3f} ms{RESET}")
    print(f"  • {BOLD}Aceleração Medida (Speedup)  :{RESET} {CYAN}{BOLD}{speedup:,.1f}x MAIS RÁPIDO!{RESET}\n")
    
    if max_dh <= 1e-14 and max_dc <= 1e-14:
        print(f"{GREEN}{BOLD}[✓] COMPROVAÇÃO CONCLUÍDA:{RESET} O resultado da GPU é matematicamente idêntico ao ordpy canônico, processado em velocidade sub-milissegundo.\n")
        return 0
    else:
        print(f"{RED}[✗] ALERTA: Desvio residual acima da tolerância esperada.{RESET}\n")
        return 1

def cmd_benchmark(args):
    """Executa um benchmark de throughput medindo bilhões de janelas por segundo."""
    print_banner()
    print(f"{BOLD}=== BENCHMARK DE DESEMPENHO E VAZÃO (THROUGHPUT) ==={RESET}\n")
    
    if not is_cuda_available():
        print(f"{RED}[ERRO] GPU CUDA indisponível.{RESET}\n")
        return 1
        
    res = args.size
    channels = args.channels
    W = args.order
    runs = args.runs
    
    tot_windows = channels * (res - W + 1) * (res - W + 1)
    print(f"[*] Configuração: {channels} canais x {res}x{res} pixels (W={W})")
    print(f"[*] Total de Janelas Deslizantes por Lote: {tot_windows:,}")
    print(f"[*] Executando {runs} repetições com sincronização estrita de hardware...\n")
    
    stack = np.random.randint(0, 256, size=(channels, res, res), dtype=np.uint8)
    
    # Warmup
    for _ in range(3):
        _ = complexity_entropy_batch(stack, dx=W, dy=W)
        
    tempos = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = complexity_entropy_batch(stack, dx=W, dy=W)
        tempos.append((time.perf_counter() - t0) * 1000.0)
        
    m_t = np.mean(tempos)
    std_t = np.std(tempos)
    throughput_giga = (tot_windows / (m_t / 1000.0)) / 1e9
    
    print(f"  {GREEN}[✓] Tempo Médio de Execução :{RESET} {BOLD}{m_t:.3f} ± {std_t:.3f} ms{RESET}")
    print(f"  {GREEN}[✓] Vazão Efetiva (Throughput):{RESET} {CYAN}{BOLD}{throughput_giga:.2f} BILHÕES de janelas/segundo{RESET}\n")
    return 0

def cmd_quickstart(args):
    """Exibe um guia rápido e interativo de uso da biblioteca em Python."""
    print_banner()
    guide = f"""{BOLD}=== GUIA RÁPIDO DE USO EM PYTHON (TUTORIAL) ==={RESET}

{YELLOW}1. Processando uma Imagem 2D (Substituição Direta do ordpy):{RESET}
```python
import numpy as np
import ordpy_gpu

# Cria imagem 2D
img = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)

# Calcula Entropia de Permutação (H) e Complexidade Estatística (C)
H, C = ordpy_gpu.complexity_entropy(img, dx=2, dy=2)
print(f"H = {{H:.6f}}, C = {{C:.6f}}")
```

{YELLOW}2. Processando Tensores Multicanal 3D (Máximo Desempenho em Lote):{RESET}
```python
import numpy as np
import ordpy_gpu

# Tensor de 13 canais cromáticos (13, 800, 800)
stack = np.random.randint(0, 256, size=(13, 800, 800), dtype=np.uint8)

# Processa todos os canais simultaneamente em sub-milissegundos
H_vec, C_vec = ordpy_gpu.complexity_entropy_batch(stack, dx=2, dy=2)
print("H por canal:", H_vec)
print("C por canal:", C_vec)
```

{YELLOW}3. Verificando a Precisão Diretamente no Código Python:{RESET}
```python
import ordpy_gpu
ordpy_gpu.verify(size=512, channels=13)
```
"""
    print(guide)
    return 0

def main():
    parser = argparse.ArgumentParser(
        prog="ordpy-gpu",
        description="Biblioteca de Alto Desempenho em GPU para Entropia de Permutação 2D e Complexidade Estatística.",
        epilog="Exemplos:\n  ordpy-gpu info\n  ordpy-gpu verify --size 800\n  ordpy-gpu benchmark --size 1024 --runs 20\n  ordpy-gpu quickstart",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis:")
    
    # Subcomando info
    sub_info = subparsers.add_parser("info", help="Exibe diagnóstico de GPU, VRAM e versões instaladas.")
    sub_info.set_defaults(func=cmd_info)
    
    # Subcomando verify
    sub_verify = subparsers.add_parser("verify", help="Audita e comprova a equivalência contra o ordpy CPU.")
    sub_verify.add_argument("--size", "-s", type=int, default=512, help="Dimensão da matriz quadrada (default: 512)")
    sub_verify.add_argument("--channels", "-c", type=int, default=13, help="Número de canais (default: 13)")
    sub_verify.add_argument("--order", "-w", type=int, default=2, choices=[2, 3], help="Ordem da vizinhança W (default: 2)")
    sub_verify.add_argument("--seed", type=int, default=42, help="Semente pseudoaleatória (default: 42)")
    sub_verify.set_defaults(func=cmd_verify)
    
    # Subcomando benchmark
    sub_bench = subparsers.add_parser("benchmark", help="Mede a vazão máxima de janelas por segundo.")
    sub_bench.add_argument("--size", "-s", type=int, default=800, help="Dimensão da matriz (default: 800)")
    sub_bench.add_argument("--channels", "-c", type=int, default=13, help="Número de canais (default: 13)")
    sub_bench.add_argument("--order", "-w", type=int, default=2, choices=[2, 3], help="Ordem da vizinhança W (default: 2)")
    sub_bench.add_argument("--runs", "-r", type=int, default=15, help="Número de repetições (default: 15)")
    sub_bench.set_defaults(func=cmd_benchmark)
    
    # Subcomando quickstart
    sub_quick = subparsers.add_parser("quickstart", help="Exibe guia rápido com exemplos de código Python.")
    sub_quick.set_defaults(func=cmd_quickstart)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
        
    args = parser.parse_args()
    if hasattr(args, "func"):
        sys.exit(args.func(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
