"""Testes unitários automatizados para ordpy-gpu usando unittest nativo.

Comprova a equivalência numérica estrita (IEEE 754 precisão dupla, tolerância 10^-14)
frente ao pacote canônico ordpy para toda a trinca de quantificadores:
- Entropia de Permutação (H)
- Complexidade Estatística (C)
- Informação de Fisher (F)
"""

import unittest
import numpy as np
import ordpy
import ordpy_gpu

class TestOrdpyGPU(unittest.TestCase):
    
    def test_01_cuda_availability(self):
        """Verifica se o backend CUDA está ativo e funcional."""
        self.assertTrue(ordpy_gpu.is_cuda_available(), "GPU CUDA deve estar disponível no ambiente de teste.")
        info = ordpy_gpu.get_device_info()
        self.assertIn("name", info)
        self.assertGreater(info.get("vram_gb", 0), 0)
        print(f"\n[OK] GPU Detectada: {info.get('name')} ({info.get('vram_gb'):.2f} GB VRAM)")

    def test_02_single_image_equivalence_w2(self):
        """Testa equivalência numérica estrita para W=2 em (H, C, F) em diferentes resoluções."""
        for res in [64, 128, 256]:
            np.random.seed(42 + res)
            img = np.random.randint(0, 256, size=(res, res), dtype=np.uint8)
            
            # Execução ordpy CPU
            h_cpu, c_cpu = ordpy.complexity_entropy(img, dx=2, dy=2)
            _, f_cpu = ordpy.fisher_shannon(img, dx=2, dy=2)
            
            # Execução ordpy-gpu trinca completa
            h_gpu, c_gpu, f_gpu = ordpy_gpu.complexity_entropy_fisher(img, dx=2, dy=2)
            
            # Tolerância de precisão dupla IEEE 754 float64 (10^-14)
            np.testing.assert_allclose(h_cpu, h_gpu, atol=1e-14, rtol=1e-14)
            np.testing.assert_allclose(c_cpu, c_gpu, atol=1e-14, rtol=1e-14)
            np.testing.assert_allclose(f_cpu, f_gpu, atol=1e-14, rtol=1e-14)
            
            # Testa interface direta fisher_shannon
            h_fs, f_fs = ordpy_gpu.fisher_shannon(img, dx=2, dy=2)
            np.testing.assert_allclose(h_cpu, h_fs, atol=1e-14)
            np.testing.assert_allclose(f_cpu, f_fs, atol=1e-14)
            
            print(f"[OK] Equivalência W=2 em {res}x{res}: Delta_H <= 1e-14, Delta_C <= 1e-14, Delta_F <= 1e-14")

    def test_03_multichannel_batch_w2(self):
        """Testa lote 3D de 13 canais cromáticos simultâneos para (H, C, F)."""
        np.random.seed(123)
        stack = np.random.randint(0, 256, size=(13, 128, 128), dtype=np.uint8)
        
        h_vec, c_vec, f_vec = ordpy_gpu.complexity_entropy_fisher_batch(stack, dx=2, dy=2)
        self.assertEqual(len(h_vec), 13)
        self.assertEqual(len(c_vec), 13)
        self.assertEqual(len(f_vec), 13)
        
        for ch in range(13):
            h_cpu, c_cpu = ordpy.complexity_entropy(stack[ch], dx=2, dy=2)
            _, f_cpu = ordpy.fisher_shannon(stack[ch], dx=2, dy=2)
            np.testing.assert_allclose(h_cpu, h_vec[ch], atol=1e-14)
            np.testing.assert_allclose(c_cpu, c_vec[ch], atol=1e-14)
            np.testing.assert_allclose(f_cpu, f_vec[ch], atol=1e-14)
        print("[OK] Lote 3D com 13 canais validado para H, C e F com sucesso!")

    def test_04_w3_execution(self):
        """Testa execução com vizinhança W=3 (362.880 estados)."""
        np.random.seed(999)
        img = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
        h_gpu, c_gpu = ordpy_gpu.complexity_entropy(img, dx=3, dy=3)
        self.assertTrue(0.0 <= h_gpu <= 1.0)
        self.assertTrue(0.0 <= c_gpu <= 1.0)
        print(f"[OK] Execução W=3 concluída: H={h_gpu:.6f}, C={c_gpu:.6f}")

if __name__ == "__main__":
    unittest.main()
