# ordpy-gpu: GPU-Accelerated 2D Permutation Entropy and Statistical Complexity via Lehmer-Ranked Inversion Counting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-CuPy-green.svg)](https://cupy.dev/)
[![Accuracy: IEEE 754](https://img.shields.io/badge/Accuracy-IEEE%20754%20(%3C10⁻¹⁵)-brightgreen.svg)](https://github.com/fabio-linhares/ordpy-gpu)
[![Peak Throughput](https://img.shields.io/badge/Throughput-57.6%20G%20windows%2Fs-blueviolet.svg)](https://github.com/fabio-linhares/ordpy-gpu)

`ordpy-gpu` é uma biblioteca de computação de alto desempenho em GPU (CUDA/C++ e Python/CuPy) para o cálculo ultrarrápido de **Entropia de Permutação Bidimensional (2D Permutation Entropy - $H$)** e **Complexidade Estatística de Rosso/Jensen–Shannon ($C$)** no Plano Complexidade–Entropia Causal (CECP).

Projetada como um substituto transparente (*drop-in replacement*) para o pacote canônico de referência (`ordpy`), a biblioteca elimina o gargalo histórico de ordenação repetitiva (`argsort`) ao reformular o mapeamento ordinal em **Códigos de Lehmer (*argsort-free*) em base fatorádica**, combinando **Fusão de Kernels (*Single-Pass*)**, **Loteamento 3D de Canais** e **Redução Entrópica por Tabelas de Busca (LUT)** diretamente na VRAM do dispositivo.

---

## 📌 Por que o `ordpy-gpu`? (Motivação e Gargalo Computacional)

A análise causal de complexidade e entropia ordinal 2D é um instrumento consagrado na análise de sistemas complexos, diagnóstico médico por imagem, sensoriamento remoto, sismologia, física de fluidos e identificação forense de imagens geradas por inteligência artificial.

Entretanto, as implementações de referência da literatura (como o pacote canônico `ordpy`, publicado na revista *Chaos* em 2021) operam sequencialmente na CPU com dois limitantes estruturais identificados por *profiling*:
1. **$\approx 60\%$ do tempo de execução** é consumido por rotinas repetitivas de ordenação comparativa (`argsort`) aplicadas individualmente sobre cada uma das centenas de milhares a milhões de janelas deslizantes locais.
2. **$\approx 40\%$ do tempo de execução** é gasto na agregação, contagem de padrões e redução de histogramas (`np.unique`).

Para matrizes de alta resolução ($800\times800$ a $4096\times4096$) ou lotes multiespectrais/multicanais, o pipeline clássico exige dezenas de segundos por imagem e horas/dias para datasets completos. O **`ordpy-gpu`** resolve esse gargalo migrando todo o fluxo para a GPU com ganhos de até cinco ordens de grandeza.

---

## 🚀 Principais Características

- **Desempenho e Aceleração Massiva:** Acelerações (*speedups*) entre **$521{,}6\times$ e $2.367{,}2\times$** em execuções padrão e até **$185.400\times$** no escalonamento multiescala frente à CPU.
- **Vazão de Pico (*Peak Throughput*):** Processamento de até **$57{,}6$ bilhões de janelas deslizantes por segundo** (em matrizes $4096\times 4096$).
- **Equivalência Numérica Estrita (IEEE 754):** Erros residuais máximos limitados a $\le 2{,}22 \times 10^{-16}$ para Entropia ($H$) e $\le 1{,}05 \times 10^{-15}$ para Complexidade ($C$).
- **API 100% Compatível com `ordpy`:** Transição transparente sem necessidade de reescrever códigos legados.
- **Loteamento Multicanal 3D Nativo:** Paralelização simultânea sobre pilhas de imagens e múltiplos canais cromáticos/espectrais (RGB, HSV, YCbCr, etc.) via grade tridimensional (`blockIdx.z`).
- **Suporte Multiescala e Ordens Espaciais:** Suporte nativo completo para $W=2$ ($k=4$, 24 classes fatorádicas) e $W=3$ ($k=9$, 362.880 classes).
- **CLI Integrada de Auditoria:** Utilitário de linha de comando (`ordpy-gpu verify`) para autodiagnóstico de hardware e validação de ponto flutuante em tempo real.

---

## 🔬 Inovação Teórica e Arquitetura do Motor CUDA

O motor do `ordpy-gpu` fundamenta-se em quatro pilares de engenharia e combinatória:

```
                                ARQUITETURA DO MOTOR ORDPY-GPU
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ GPU GLOBAL MEMORY (VRAM)                                                               │
│  Input Matrix / 3D Tensor Stack (B, H, W) [Uint8 / Float32 / Float64]                  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Single-Pass Read
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ SINGLE-PASS FUSED CUDA KERNEL (1 Thread = 1 Sliding Window)                            │
│  ├─ 1. Extração da Janela Local WxW (Sem alocação intermediária em VRAM)               │
│  ├─ 2. Mapeamento Ordinal Branchless O(k²) via Códigos de Lehmer (Base Fatorádica)     │
│  │     * Bijeção estrita e desempate estável garantidos pelo Teorema 1                 │
│  └─ 3. Acumulação Atômica no Histograma de Padrões (Shared Memory / Global VRAM)       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Atomic Aggregation
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ DEVICE-SIDE REDUCTION KERNEL (LUT Logarithms & Complexity Constants)                   │
│  ├─ Shannon Permutation Entropy (H) via Fast Device Lookup Tables                      │
│  └─ Jensen-Shannon Statistical Complexity (C) Reduction (Zero CPU Roundtrip)           │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Final Scalars
                                            ▼
                                  Output: (H, C) Tuple
```

1. **Códigos de Lehmer (*Argsort-Free*):** Mapeamento bijetivo unívoco da permutação ordinal para a base fatorádica $[0, k!-1]$ mediante contagem estática de inversões à direita em $O(k^2)$ operações puramente relacionais *branchless*.
2. **Teorema 1 (Bijeção e Desempate Estável):** Prova matemática formal que garante que a contagem de inversões estáticas com desigualdade estrita preserva exatamente a convenção de desempate estável (*stable sort*) do `argsort` clássico sob valores de pixel idênticos.
3. **Fusão de Kernels (*Kernel Fusion*):** Eliminação completa da materialização de matrizes ordinais intermediárias na memória global da GPU.
4. **Redução Direta no Device por LUT:** Redução do histograma e cálculo logarítmico realizados inteiramente na GPU utilizando tabelas pré-computadas (*Look-Up Tables*), eliminando transferências intermediárias *Device-to-Host*.

---

## 📊 Estudo de Ablação e Benchmarks

### 1. Estudo de Ablação dos Mecanismos de Ganho (Imagem $800 \times 800$, 13 canais)

| Nível Arquitetural | Descrição da Abordagem | Tempo Médio | Speedup vs. CPU | Ganho Incremental |
| :--- | :--- | :---: | :---: | :---: |
| **N1: CPU Canônica (`ordpy`)** | Sequencial `argsort` + `np.unique` | $30.410\text{ ms}$ | $1{,}0\times$ | Linha de Base |
| **N2: GPU Convencional** | Paralelização ingênua com Sort em GPU | $1.010\text{ ms}$ | $30{,}1\times$ | $30{,}1\times$ |
| **N3: GPU Lehmer** | Substituição por Códigos de Lehmer | $505\text{ ms}$ | $60{,}2\times$ | $\approx 2{,}0\times$ |
| **N4: GPU Full (`ordpy-gpu`)** | Lehmer + *Kernel Fusion* + Lote 3D + LUT | $\mathbf{0{,}24\text{ ms}}$ | $\mathbf{127.389\times}$ | $\mathbf{12{,}2\times}$ |

### 2. Escalabilidade Multiescala por Resolução (GPU NVIDIA RTX 4070 Laptop, 13 Canais, $W=2$)

| Resolução | Janelas ($B \times M$) | CPU (`ordpy`) | `ordpy-gpu` | Speedup Global | Vazão de Janelas |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$128 \times 128$** | $209.677$ | $0{,}59\text{ s}$ | $\mathbf{0{,}15\text{ ms}}$ | **$3.953\times$** | $17{,}1\text{ G janelas/s}$ |
| **$256 \times 256$** | $845.325$ | $2{,}54\text{ s}$ | $\mathbf{0{,}17\text{ ms}}$ | **$14.842\times$** | $37{,}1\text{ G janelas/s}$ |
| **$512 \times 512$** | $3.394.573$ | $11{,}60\text{ s}$ | $\mathbf{0{,}18\text{ ms}}$ | **$65.603\times$** | $52{,}8\text{ G janelas/s}$ |
| **$800 \times 800$** | $8.299.213$ | $30{,}41\text{ s}$ | $\mathbf{0{,}24\text{ ms}}$ | **$127.389\times$** | **$56{,}2\text{ G janelas/s}$** |
| **$1024 \times 1024$** | $13.604.877$ | $\approx 50\text{ s}$ | $\mathbf{0{,}24\text{ ms}}$ | **$> 200.000\times$** | **$57{,}6\text{ G janelas/s}$** |
| **$4096 \times 4096$** | $217.997.325$ | $\approx 15\text{ min}$ | $\mathbf{5{,}19\text{ ms}}$ | --- | $42{,}0\text{ G janelas/s}$ |

### 3. Fidelidade Numérica estrita perante a CPU (IEEE 754)

| Métrica | CPU Canônica (`ordpy`) | GPU Proposta (`ordpy-gpu`) | Erro Residual Máximo | Status de Auditoria |
| :--- | :---: | :---: | :---: | :---: |
| **Entropia Normalizada ($H$)** | $0{,}997194605928$ | $0{,}997194605928$ | $\mathbf{2{,}22 \times 10^{-16}}$ | ✅ Exata (Zero Desvio) |
| **Complexidade Estatística ($C$)** | $0{,}004456673551$ | $0{,}004456673551$ | $\mathbf{1{,}05 \times 10^{-15}}$ | ✅ Exata (Limite Físico) |

---

## 📦 Instalação

### Pré-requisitos
- GPU NVIDIA com suporte a CUDA (Compute Capability $\ge 6.0$).
- Python $\ge 3.8$.
- Drivers NVIDIA e CUDA Toolkit (11.x ou 12.x).

### Instalação via pip

```bash
# Instalação direta do repositório
pip install git+https://github.com/fabio-linhares/ordpy-gpu.git

# Ou clone local em modo editável
git clone https://github.com/fabio-linhares/ordpy-gpu.git
cd ordpy-gpu
pip install -e .
```

Se o CuPy correspondente à sua versão do CUDA não estiver instalado:
```bash
pip install cupy-cuda12x  # Para CUDA 12.x
# ou
pip install cupy-cuda11x  # Para CUDA 11.x
```

---

## 💡 Guia Rápido de Uso

### 1. Imagem Individual 2D (Substituição Transparente do `ordpy`)

```python
import numpy as np
import ordpy_gpu

# Matriz de entrada (Uint8 ou Float)
img = np.random.randint(0, 256, size=(800, 800), dtype=np.uint8)

# Cálculo com interface idêntica ao ordpy
H, C = ordpy_gpu.complexity_entropy(img, dx=2, dy=2)

print(f"Entropia de Permutação (H) : {H:.8f}")
print(f"Complexidade Estatística (C): {C:.8f}")
```

### 2. Lote Multicanal 3D (Máximo Desempenho em Lotes)

```python
import numpy as np
import ordpy_gpu

# Pilha tridimensional de 13 canais cromáticos (13, 800, 800)
image_stack = np.random.randint(0, 256, size=(13, 800, 800), dtype=np.uint8)

# Processa todos os canais simultaneamente no mesmo lançamento de Kernel CUDA
H_vec, C_vec = ordpy_gpu.complexity_entropy_batch(image_stack, dx=2, dy=2)

for i, (h, c) in enumerate(zip(H_vec, C_vec)):
    print(f"Canal {i:02d} -> H: {h:.6f} | C: {c:.6f}")
```

---

## 🛠️ Utilitário de Linha de Comando (CLI)

O pacote fornece uma ferramenta nativa de terminal para diagnóstico, auditoria matemática e calibração de hardware:

```bash
# Diagnóstico de Hardware, Dispositivo CUDA e Versões
ordpy-gpu info

# Auditoria de Equivalência Numérica Estrita (GPU vs ordpy CPU)
ordpy-gpu verify --size 512 --channels 13

# Medição de Throughput de Pico e Latência Média
ordpy-gpu benchmark --size 800 --channels 13 --runs 20

# Guia Rápido Interativo no Terminal
ordpy-gpu quickstart
```

---

## 🧪 Testes Automatizados e Integração Contínua

Para executar a suíte completa de testes de regressão e auditoria IEEE 754:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 📚 Citação Acadêmica

Se você utilizar a biblioteca `ordpy-gpu` ou seus conceitos algorítmicos em seus trabalhos científicos, por favor cite:

```bibtex
@article{ordpy_gpu_2026,
  title={GPU-Accelerated 2D Permutation Entropy and Statistical Complexity via Lehmer-Ranked Inversion Counting},
  author={Linhares, Fabio and Nogueira, Bruno Costa and Pinheiro, Rian Gabriel Santos and Queiroz, Fabiane da Silva},
  journal={Instituto de Computacao (IC), Universidade Federal de Alagoas (UFAL)},
  year={2026},
  url={https://github.com/fabio-linhares/ordpy-gpu}
}

@software{ordpy_gpu_software_2026,
  author={Linhares, Fabio and Nogueira, Bruno Costa and Pinheiro, Rian Gabriel Santos and Queiroz, Fabiane da Silva},
  title={ordpy-gpu: GPU-Accelerated 2D Permutation Entropy and Statistical Complexity in CUDA},
  year={2026},
  publisher={GitHub},
  url={https://github.com/fabio-linhares/ordpy-gpu},
  version={1.0.0}
}
```

---

## 📄 Licença

Distribuído sob a licença [MIT](LICENSE).
