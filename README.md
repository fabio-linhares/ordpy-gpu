# ordpy-gpu: GPU-Accelerated 2D Permutation Entropy, Statistical Complexity and Fisher Information via Lehmer-Ranked Inversion Counting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-CuPy-green.svg)](https://cupy.dev/)
[![Accuracy: IEEE 754](https://img.shields.io/badge/Accuracy-IEEE%20754%20(%3C10⁻¹⁵)-brightgreen.svg)](https://github.com/fabio-linhares/ordpy-gpu)
[![Peak Throughput](https://img.shields.io/badge/Throughput-57.6%20G%20windows%2Fs-blueviolet.svg)](https://github.com/fabio-linhares/ordpy-gpu)

`ordpy-gpu` é uma biblioteca de computação de alto desempenho em GPU (CUDA/C++ e Python/CuPy) para o cálculo ultrarrápido de **Entropia de Permutação Bidimensional (2D Permutation Entropy - $H$)**, **Complexidade Estatística de Rosso/Jensen–Shannon ($C$)** e **Informação de Fisher Discreta ($F$)** nos Planos Causais Complexidade–Entropia (CECP) e Fisher–Shannon (CFSP).

Projetada como um substituto transparente (*drop-in replacement*) para a biblioteca de referência (`ordpy`), a biblioteca elimina o gargalo histórico de ordenação repetitiva (`argsort`) ao reformular o mapeamento ordinal em **Códigos de Lehmer (*argsort-free*) em base fatorádica**, combinando **Fusão de Kernels (*Single-Pass*)**, **Loteamento 3D de Canais** e **Redução Entrópica por Tabelas de Busca (LUT)** diretamente na VRAM do dispositivo.

---

## 📌 Por que o `ordpy-gpu`? (Motivação e Gargalo Computacional)

A análise causal de complexidade e entropia ordinal 2D é um instrumento consagrado na análise de sistemas complexos, diagnóstico médico por imagem, sensoriamento remoto, sismologia, física de fluidos e identificação forense de imagens geradas por inteligência artificial.

Entretanto, as implementações de referência da literatura (como o pacote `ordpy`, publicado na revista *Chaos* em 2021) operam sequencialmente na CPU com dois limitantes estruturais identificados por *profiling*:
1. **$\approx 60\%$ do tempo de execução** é consumido por rotinas repetitivas de ordenação comparativa (`argsort`) aplicadas individualmente sobre cada uma das centenas de milhares a milhões de janelas deslizantes locais.
2. **$\approx 40\%$ do tempo de execução** é gasto na agregação, contagem de padrões e redução de histogramas (`np.unique`).

Para matrizes de alta resolução ($800\times800$ a $4096\times4096$) ou lotes multiespectrais/multicanais, o pipeline clássico exige dezenas de segundos por imagem e horas/dias para datasets completos. O **`ordpy-gpu`** resolve esse gargalo migrando todo o fluxo para a GPU com ganhos de até cinco ordens de grandeza.

---

## 🚀 Principais Características

- **Arquitetura Paralela Completa em GPU:** A biblioteca processa múltiplas camadas em paralelo (`blockIdx.z`), suporta dimensões de incorporação **quadradas e retangulares** ($D_x \times D_y$ com $2 \le D_x D_y \le 9$) e executa a redução entrópica integralmente no dispositivo por tabelas de busca (LUT), retornando ao hospedeiro apenas os valores finais de entropia e complexidade.
- **Desempenho e Aceleração Massiva:** Em lotes multicanal, o ganho de velocidade alcançou até **cinco ordens de magnitude** ($127.389\times$ na RTX 4070 em $800\times 800$) em arquiteturas de entrada (MX150), alto desempenho (RTX 4070) e *datacenter* (NVIDIA A100-SXM4-80GB), além de **$1.510\times$ a $2.135\times$** em configurações retangulares ($D_x \neq D_y$).
- **Eficiência de *Warps* Comprovada no Silício:** *Profiling* arquitetural via NVIDIA Nsight Compute (`ncu` 2026.4.0) na RTX 4070 sobre imagens real (RAISE) e sintética (Synthbuster) comprovou **$99{,}8\%$ de *threads* ativas por *warp*** ($31{,}94/32$), redução de $22 \to 16$ registradores/*thread* e **$98{,}97\%$ de taxa de acerto em cache L2**.
- **Vazão de Pico (*Peak Throughput*):** Processamento de até **$57{,}6$ bilhões de janelas/s** em $1024\times 1024$ (RTX 4070, $2\times 2$) e execução ponta a ponta de $4096\times 4096$ com $3\times 3$ ($362.880$ estados) em **$64{,}81\text{ ms}$** na NVIDIA A100-SXM4-80GB.
- **Equivalência Numérica Estrita (IEEE 754):** Nos experimentos $2\times 2$, reproduziu a referência `ordpy` em CPU com desvios máximos de **$2{,}22\times 10^{-16}$** na entropia ($H$) e **$1{,}05\times 10^{-15}$** na complexidade ($C$), e $\le 1{,}15\times 10^{-14}$ em janelas retangulares ($D_x \neq D_y$).
- **API 100% Compatível com `ordpy`:** Suporte a `dx, dy` arbitrários ($2 \le D_x D_y \le 9$), Informação de Fisher Discreta ($F$) e utilitário CLI de auditoria (`ordpy-gpu verify`).

---

## 🔬 Inovação Teórica e Arquitetura do Motor CUDA

O motor do `ordpy-gpu` fundamenta-se em quatro pilares de engenharia e combinatória:

```
                                ARQUITETURA DO MOTOR ORDPY-GPU
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ GPU GLOBAL MEMORY (VRAM)                                                               │
│  Input Matrix / 3D Tensor Stack (B, Ny, Nx) [Uint8 / Float32 / Float64]                │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Single-Pass Read
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ SINGLE-PASS FUSED CUDA KERNEL (1 Thread = 1 Sliding Window)                            │
│  ├─ 1. Extração da Janela Local Dx x Dy (Sem alocação intermediária em VRAM)           │
│  ├─ 2. Mapeamento Ordinal Branchless O(k²) via Códigos de Lehmer (Base Fatorádica)     │
│  │     * Bijeção estrita e desempate estável garantidos pelo Teorema 1                 │
│  └─ 3. Acumulação Atômica: Shared Memory (k=4, 96 B) ou Global Esparsa (4 < k <= 9)    │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Atomic Aggregation
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ DEVICE-SIDE REDUCTION KERNEL (LUT Logarithms & Complexity Constants)                   │
│  ├─ Shannon Permutation Entropy (H) via Fast Device Lookup Tables                      │
│  └─ Jensen-Shannon Statistical Complexity (C) Reduction (Zero CPU Roundtrip)           │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Final Scalars (B pairs)
                                            ▼
                                  Output: (H, C) Tuple
```

1. **Códigos de Lehmer (*Argsort-Free*):** Mapeamento bijetivo unívoco da permutação ordinal para a base fatorádica $[0, k!-1]$ mediante contagem estática de inversões à direita em $O(k^2)$ operações puramente relacionais *branchless*.
2. **Teorema 1 (Bijeção e Desempate Estável):** Prova matemática formal que garante que a contagem de inversões estáticas com desigualdade estrita $[v_j < v_i]$ ($j > i$) corresponde ao código de Lehmer da permutação inversa $\sigma(V)^{-1}$, preservando a ordenação estável sob empates.
3. **Fusão de Kernels (*Kernel Fusion*):** Eliminação completa da materialização de matrizes ordinais intermediárias na memória global da GPU.
4. **Redução Direta no Device por LUT:** Redução do histograma e cálculo logarítmico realizados inteiramente na GPU utilizando tabelas pré-computadas (*Look-Up Tables*), retornando ao hospedeiro apenas os $B$ pares finais $(H, C)$.

---

## 📊 Estudo de Ablação, Profiling Nsight e Benchmarks Comprobatórios

Todos os dados brutos de reprodutibilidade estão disponíveis na pasta [`benchmarks/`](benchmarks/).

### 1. Estudo de Ablação em 4 Níveis (RTX 4070 Laptop, 13 canais, $2\times 2$)
Fonte: [`benchmarks/ablation_study_results.csv`](benchmarks/ablation_study_results.csv)

| Resolução | N1: CPU (`ordpy`) | N2: GPU-Sort | N3: GPU-Lehmer | N4: GPU-Full | Ganho Lehmer (N2→N3) | Ganho Fusão/LUT (N3→N4) | Speedup Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$128\times 128$** | $0{,}5905\text{ s}$ | $0{,}361\text{ ms}$ | $0{,}360\text{ ms}$ | **$0{,}149\text{ ms}$** | $1{,}00\times$ | $2{,}41\times$ | **$3.954\times$** |
| **$256\times 256$** | $2{,}5417\text{ s}$ | $0{,}826\text{ ms}$ | $0{,}562\text{ ms}$ | **$0{,}171\text{ ms}$** | $1{,}47\times$ | $3{,}28\times$ | **$14.843\times$** |
| **$512\times 512$** | $11{,}5967\text{ s}$ | $3{,}185\text{ ms}$ | $1{,}445\text{ ms}$ | **$0{,}177\text{ ms}$** | $2{,}20\times$ | $8{,}18\times$ | **$65.603\times$** |
| **$800\times 800$** | $30{,}4060\text{ s}$ | $5{,}569\text{ ms}$ | $2{,}922\text{ ms}$ | **$0{,}239\text{ ms}$** | $1{,}91\times$ | $12{,}24\times$ | **$127.389\times$** |

### 2. Profiling Arquitetural via NVIDIA Nsight Compute (`ncu` 2026.4.0, RTX 4070, $800\times 800$, Real RAISE + Sintética Synthbuster)
Fonte: [`benchmarks/nsight_ncu_rtx4070.csv`](benchmarks/nsight_ncu_rtx4070.csv)

| Núcleo Avaliado | Reg./thread | Mem. Comp. | Threads Ativas/Warp | SM Throughput | L1 Hit Rate | L2 Hit Rate | Mem. Throughput | Achieved Occ. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **N2: `GPU-Sort`** ($1$ canal, $2\times 2$) | $22$ | $0\text{ B}$ | $27{,}94 / 32$ ($87{,}3\%$) | $9{,}15\%$ | $87{,}76\%$ | $99{,}41\%$ | $1{,}08\text{ GB/s}$ | $90{,}52\%$ |
| **N3: `GPU-Lehmer`** ($1$ canal, $2\times 2$) | $16$ | $0\text{ B}$ | **$31{,}94 / 32$ ($99{,}8\%$)** | $2{,}20\%$ | $30{,}62\%$ | $93{,}55\%$ | $2{,}25\text{ GB/s}$ | $80{,}66\%$ |
| **N4: `Fused 3D`** ($26$ canais, $2\times 2$) | $16$ | $96\text{ B}$ | $31{,}76 / 32$ ($99{,}3\%$) | **$56{,}69\%$** | $0{,}03\%$ | **$98{,}97\%$** | $0{,}04\text{ GB/s}$ | $57{,}38\%$ |
| **N4: `Fused 3D`** ($26$ canais, $3\times 3$) | $22$ | $0\text{ B}$ | **$31{,}98 / 32$ ($99{,}9\%$)** | $47{,}64\%$ | $60{,}00\%$ | $86{,}87\%$ | $0{,}08\text{ GB/s}$ | $34{,}78\%$ |

### 3. Dimensões de Incorporação Retangulares ($D_x \neq D_y$, $800\times 800$: RTX 4070 $13$ch e A100 $26$ch reais+sintéticos vs. `ordpy` CPU)
Fontes: [`benchmarks/benchmark_retangular_dx_dy.json`](benchmarks/benchmark_retangular_dx_dy.json) e [`benchmarks/colab_benchmark_datacenter_gpu_executado_v2.ipynb`](benchmarks/colab_benchmark_datacenter_gpu_executado_v2.ipynb)

| $D_x \times D_y$ | $k$ | Estados ($k!$) | CPU (`ordpy`) | RTX 4070 ($13$ch) | A100 Colab ($26$ch) | Speedup (RTX) | $\max|\Delta H|$ | $\max|\Delta C|$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$1\times 4$** | $4$ | $24$ | $19{,}2147\text{ s}$ | $11{,}04 \pm 0{,}90\text{ ms}$ | $13{,}35 \pm 0{,}09\text{ ms}$ | **$1.740{,}1\times$** | $2{,}22\times 10^{-16}$ | $7{,}33\times 10^{-16}$ |
| **$4\times 1$** | $4$ | $24$ | $18{,}1771\text{ s}$ | $12{,}04 \pm 0{,}98\text{ ms}$ | $10{,}50 \pm 0{,}38\text{ ms}$ | **$1.510{,}0\times$** | $1{,}11\times 10^{-16}$ | $1{,}47\times 10^{-15}$ |
| **$2\times 3$** | $6$ | $720$ | $21{,}0945\text{ s}$ | $11{,}82 \pm 0{,}74\text{ ms}$ | $7{,}85 \pm 0{,}08\text{ ms}$ | **$1.785{,}0\times$** | $3{,}33\times 10^{-16}$ | $1{,}29\times 10^{-15}$ |
| **$3\times 2$** | $6$ | $720$ | $21{,}4840\text{ s}$ | $11{,}71 \pm 0{,}91\text{ ms}$ | $7{,}89 \pm 0{,}07\text{ ms}$ | **$1.835{,}2\times$** | $3{,}33\times 10^{-16}$ | $1{,}94\times 10^{-15}$ |
| **$2\times 4$** | $8$ | $40.320$ | $22{,}5958\text{ s}$ | $10{,}73 \pm 0{,}75\text{ ms}$ | $6{,}08 \pm 0{,}13\text{ ms}$ | **$2.105{,}5\times$** | $8{,}88\times 10^{-16}$ | $7{,}67\times 10^{-15}$ |
| **$4\times 2$** | $8$ | $40.320$ | $22{,}2288\text{ s}$ | $10{,}41 \pm 0{,}75\text{ ms}$ | $5{,}61 \pm 0{,}09\text{ ms}$ | **$2.135{,}3\times$** | $8{,}88\times 10^{-16}$ | $1{,}15\times 10^{-14}$ |

### 4. Escalonamento em Larga Escala na RTX 4070 Laptop ($13$ canais, $2\times 2$, $15$ repetições)
Fonte: [`benchmarks/large_scale_benchmark_results.csv`](benchmarks/large_scale_benchmark_results.csv)

| Resolução | Janelas ($B \cdot M$) | `ordpy-gpu` ($2\times 2$) | Vazão Efetiva ($2\times 2$) |
| :---: | :---: | :---: | :---: |
| **$128 \times 128$** | $209.677$ | $0{,}0122 \pm 0{,}0003\text{ ms}$ | $17{,}1\text{ G janelas/s}$ |
| **$256 \times 256$** | $845.325$ | $0{,}0228 \pm 0{,}0006\text{ ms}$ | $37{,}1\text{ G janelas/s}$ |
| **$512 \times 512$** | $3.394.573$ | $0{,}0643 \pm 0{,}0002\text{ ms}$ | $52{,}8\text{ G janelas/s}$ |
| **$800 \times 800$** | $8.299.213$ | $0{,}1477 \pm 0{,}0003\text{ ms}$ | $56{,}2\text{ G janelas/s}$ |
| **$1024 \times 1024$** | $13.604.877$ | $0{,}2361 \pm 0{,}0002\text{ ms}$ | **$57{,}6\text{ G janelas/s}$ (Pico)** |
| **$2048 \times 2048$** | $54.472.717$ | $1{,}0221 \pm 0{,}0007\text{ ms}$ | $53{,}3\text{ G janelas/s}$ |
| **$4096 \times 4096$** | $217.997.325$ | $5{,}1876 \pm 0{,}5736\text{ ms}$ | $42{,}0\text{ G janelas/s}$ |

### 5. Portabilidade em GPU Datacenter: NVIDIA A100-SXM4-80GB (Ponta a Ponta, $13$ canais, IC 95%)
Fonte: [`benchmarks/colab_benchmark_datacenter_gpu_executado.ipynb`](benchmarks/colab_benchmark_datacenter_gpu_executado.ipynb) e [`benchmarks/colab_benchmark_datacenter_gpu_executado_v2.ipynb`](benchmarks/colab_benchmark_datacenter_gpu_executado_v2.ipynb) (Corpus Real RAISE + Sintético DiffusionDB, $250$ imagens / $3.250$ planos) | Paridade real+sintética: $\max|\Delta H| \le 2{,}22\times 10^{-16}$ ($2\times 2$, $2\times 3$) e $3{,}89\times 10^{-15}$ ($3\times 3$); $\max|\Delta C| \le 1{,}30\times 10^{-15}$ ($2\times 2$) e $1{,}49\times 10^{-13}$ ($3\times 3$). Estresse $4096\times 4096$ derivado de foto real RAISE ($13$ch): $62{,}69 \pm 0{,}07\text{ ms}$ ($2\times 2$) e $88{,}48 \pm 0{,}39\text{ ms}$ ($3\times 3$).

| Janela | Resolução | Janelas ($B \cdot M$) | Tempo Ponta a Ponta $\pm$ IC 95% | Vazão Ponta a Ponta |
| :---: | :---: | :---: | :---: | :---: |
| **$2\times 2$** | $128\times 128$ | $209.677$ | $0{,}5495 \pm 0{,}0166\text{ ms}$ | $0{,}382\text{ G janelas/s}$ |
| **$2\times 2$** | $256\times 256$ | $845.325$ | $0{,}6272 \pm 0{,}0093\text{ ms}$ | $1{,}348\text{ G janelas/s}$ |
| **$2\times 2$** | $512\times 512$ | $3.394.573$ | $0{,}8476 \pm 0{,}0165\text{ ms}$ | $4{,}005\text{ G janelas/s}$ |
| **$2\times 2$** | $800\times 800$ | $8.299.213$ | $1{,}6939 \pm 0{,}0599\text{ ms}$ | **$4{,}899\text{ G janelas/s}$** |
| **$2\times 2$** | $1024\times 1024$ | $13.604.877$ | $2{,}9463 \pm 0{,}1850\text{ ms}$ | $4{,}618\text{ G janelas/s}$ |
| **$2\times 2$** | $2048\times 2048$ | $54.472.717$ | $15{,}5170 \pm 0{,}0229\text{ ms}$ | $3{,}511\text{ G janelas/s}$ |
| **$2\times 2$** | $4096\times 4096$ | $217.997.325$ | $62{,}0538 \pm 0{,}1264\text{ ms}$ | $3{,}513\text{ G janelas/s}$ |
| **$3\times 3$** | $800\times 800$ | $8.278.452$ | $2{,}2139 \pm 0{,}1401\text{ ms}$ | $3{,}739\text{ G janelas/s}$ |
| **$3\times 3$** | $1024\times 1024$ | $13.578.292$ | $3{,}4768 \pm 0{,}1948\text{ ms}$ | $3{,}905\text{ G janelas/s}$ |
| **$3\times 3$** | $2048\times 2048$ | $54.419.508$ | $16{,}4956 \pm 0{,}0226\text{ ms}$ | $3{,}299\text{ G janelas/s}$ |
| **$3\times 3$** | $4096\times 4096$ | $217.890.868$ | **$64{,}8050 \pm 0{,}0661\text{ ms}$** | $3{,}362\text{ G janelas/s}$ |
| **$2\times 2$** | $800\times 800$ (250 imgs) | $2.074.803.200$ | **$584{,}2300 \pm 1{,}2800\text{ ms}$** | $3{,}550\text{ G janelas/s}$ |
| **$3\times 3$** | $800\times 800$ (250 imgs) | $2.069.613.000$ | **$836{,}1200 \pm 1{,}0000\text{ ms}$** | $2{,}475\text{ G janelas/s}$ |

### 6. Fidelidade Numérica Estrita perante a CPU (IEEE 754, $800\times 800$, 13 camadas)

| Métrica | CPU de Referência (`ordpy`) | GPU Proposta (`ordpy-gpu`) | Erro Residual Máximo | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Entropia Normalizada ($H$)** | $0{,}997194605928$ | $0{,}997194605928$ | $\mathbf{2{,}22 \times 10^{-16}}$ | ✅ Exata |
| **Complexidade Estatística ($C$)** | $0{,}004456673551$ | $0{,}004456673551$ | $\mathbf{1{,}05 \times 10^{-15}}$ | ✅ Exata |

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

### 3. Trinca Completa com Informação de Fisher (H, C, F) e Plano Fisher–Shannon

```python
import numpy as np
import ordpy_gpu

img = np.random.randint(0, 256, size=(800, 800), dtype=np.uint8)

# Trinca completa (H, C, F) em GPU via códigos de Lehmer
H, C, F = ordpy_gpu.complexity_entropy_fisher(img, dx=2, dy=2)
print(f"Entropia (H)     : {H:.8f}")
print(f"Complexidade (C) : {C:.8f}")
print(f"Fisher (F)       : {F:.8f}")

# Ou interface compatível com ordpy.fisher_shannon
H, F = ordpy_gpu.fisher_shannon(img, dx=2, dy=2)

# Lote multicanal 3D com a trinca (H, C, F)
H_arr, C_arr, F_arr = ordpy_gpu.complexity_entropy_fisher_batch(image_stack, dx=2, dy=2)
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
