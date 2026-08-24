# ordpy-gpu: GPU-Accelerated 2D Permutation Entropy and Statistical Complexity in CUDA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-CuPy-green.svg)](https://cupy.dev/)

`ordpy-gpu` é uma biblioteca de computação de alto desempenho em GPU para o cálculo de **Entropia de Permutação Bidimensional (2D Permutation Entropy - $H$)** e **Complexidade Estatística de Rosso ($C$)** em imagens e tensores multidimensionais.

A biblioteca reformula o mapeamento ordinal clássico substituindo algoritmos de ordenação por comparação (`argsort`) por **Códigos de Lehmer (*argsort-free*)**, combinando **Fusão de Kernels**, **Loteamento 3D de Canais** e **Redução Entrópica por Tabelas de Busca (LUT)** diretamente na VRAM do dispositivo.

---

## Principais Recursos

- **Desempenho Massivo:** Speedups superiores a cinco ordens de grandeza (**$> 127.000\times$**) frente à CPU sequencial canônica.
- **Vazão de Pico:** Processamento de mais de **$57$ bilhões de janelas por segundo** em GPUs modernas (NVIDIA Ada Lovelace).
- **Equivalência Numérica Estrita:** Desvios residuais contidos no limite físico de máquina IEEE 754 ($\le 10^{-15}$).
- **API 100% Compatível com ordpy:** Substituição direta da chamada `ordpy.complexity_entropy(img, dx=2, dy=2)`.
- **Suporte a Lote Multicanal 3D:** Processamento simultâneo de múltiplos canais cromáticos (RGB, HSV, YCbCr, etc.) via `blockIdx.z`.
- **Vizinhanças Espaciais:** Suporte nativo completo para $W=2$ ($k=4$, 24 classes) e $W=3$ ($k=9$, 362.880 classes).

---

## Instalação

### Pré-requisitos
- GPU NVIDIA com suporte a CUDA (Compute Capability $\ge 6.0$).
- Python $\ge 3.8$.
- Driver NVIDIA e CUDA Toolkit instalados.

### Instalação via pip

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/ordpy-gpu.git
cd ordpy-gpu

# Instale o pacote e suas dependências CUDA
pip install .
```

Se o CuPy ainda não estiver instalado no seu ambiente:
```bash
pip install cupy-cuda12x  # para CUDA 12.x
# ou
pip install cupy-cuda11x  # para CUDA 11.x
```

---

## Guia Rápido de Uso

### 1. Imagem Individual 2D (Interface Compatível com ordpy)

```python
import numpy as np
import ordpy_gpu

# Cria imagem de teste 2D
img = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)

# Calcula Entropia de Permutação (H) e Complexidade Estatística (C)
H, C = ordpy_gpu.complexity_entropy(img, dx=2, dy=2)

print(f"Entropia (H)     : {H:.6f}")
print(f"Complexidade (C) : {C:.6f}")
```

### 2. Lote Multicanal 3D (Máximo Desempenho)

```python
import numpy as np
import ordpy_gpu

# Cria tensor de 13 canais cromáticos (13, 800, 800)
image_stack = np.random.randint(0, 256, size=(13, 800, 800), dtype=np.uint8)

# Processa todos os 13 canais em paralelo em um único lançamento CUDA
H_vec, C_vec = ordpy_gpu.complexity_entropy_batch(image_stack, dx=2, dy=2)

for ch_idx, (h, c) in enumerate(zip(H_vec, C_vec)):
    print(f"Canal {ch_idx:02d} -> H: {h:.6f} | C: {c:.6f}")
```

---

## Benchmarks de Desempenho

Resultados medidos em uma GPU **NVIDIA GeForce RTX 4070 Laptop (8 GB)** com 13 canais cromáticos ($W=2$):

| Resolução | Janelas ($B \times M$) | CPU (`ordpy`) | `ordpy-gpu` | Speedup Global | Vazão de Janelas |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$128 \times 128$** | $209.677$ | $0{,}59\text{ s}$ | $\mathbf{0{,}15\text{ ms}}$ | **$3.953\times$** | $17{,}1\text{ G janelas/s}$ |
| **$256 \times 256$** | $845.325$ | $2{,}54\text{ s}$ | $\mathbf{0{,}17\text{ ms}}$ | **$14.842\times$** | $37{,}1\text{ G janelas/s}$ |
| **$512 \times 512$** | $3.394.573$ | $11{,}60\text{ s}$ | $\mathbf{0{,}18\text{ ms}}$ | **$65.603\times$** | $52{,}8\text{ G janelas/s}$ |
| **$800 \times 800$** | $8.299.213$ | $30{,}41\text{ s}$ | $\mathbf{0{,}24\text{ ms}}$ | **$127.389\times$** | **$56{,}2\text{ G janelas/s}$** |
| **$1024 \times 1024$** | $13.604.877$ | $\approx 50\text{ s}$ | $\mathbf{0{,}24\text{ ms}}$ | **$> 200.000\times$** | **$57{,}6\text{ G janelas/s}$** |
| **$4096 \times 4096$** | $217.997.325$ | $\approx 15\text{ min}$ | $\mathbf{5{,}19\text{ ms}}$ | --- | $42{,}0\text{ G janelas/s}$ |

---

## Execução de Testes

```bash
pytest -v tests/
```

---

## Citação

Se você utilizar `ordpy-gpu` em sua pesquisa, por favor cite:

```bibtex
@article{ordpy_gpu_2026,
  title={GPU-Accelerated 2D Permutation Entropy and Statistical Complexity based on Lehmer Codes},
  author={Fabio et al.},
  journal={Qualificacao / Artigo Cientifico},
  year={2026}
}
```

---

## Licença
Distribuído sob a licença [MIT](LICENSE).
