# Bring-up Log

## 2026-04-19 — Repository initialization

- Created project structure based on upstream DeepGEMM analysis
- Identified SM120e blockers: 49 KB shared memory (vs 168 KB), missing `tcgen05.fence`
- Documented architecture comparison in `docs/compatibility.md`
- Proposed tile sizes: M=64, N=64, K=32 (20 KB) or K=64 (24 KB) for 49 KB smem
- Prior art review: Issue #236, createthis fork (barrier issue at 46 KB smem)

### Key References
- Upstream: https://github.com/deepseek-ai/DeepGEMM (public-release-260416)
- SM120 issue: https://github.com/deepseek-ai/DeepGEMM/issues/236
- Attempted port: https://github.com/createthis/DeepGEMM/pull/1

### Hardware Available
- 7x RTX PRO 6000 Blackwell (SM120, 96 GB each)
- CUDA 13.0, Driver 580.126.09
- GPU 0 designated for kernel development/testing

### Next Steps
1. Run `scripts/bootstrap.sh` to clone upstream and set up dev env
2. Run `scripts/test_sm120e_env.sh` to verify environment
3. Study `deep_gemm/impls/` for shared memory and barrier patterns
4. Start with `fp8_gemm_nt` kernel port

## 2026-04-19 — Environment verification complete

### test_sm120e_env.sh results
```
GPU:              RTX PRO 6000 Blackwell (SM120, compute 12.0)
Shared memory:    49,152 bytes (48 KB)
VRAM:             102 GB
PyTorch:          2.11.0+cu130
CUTLASS:          found
FP8 E4M3:        supported
nvcc:             not found (CUDA toolkit not installed system-wide)
Driver:           580.126.09
```

### GPU allocation
- GPU 0: dedicated to DeepGEMM development (96 GB free)
- GPU 1-6: DFlash hidden state collection running (6 workers, ~1,500 samples collected)

### Observations
- `shared_memory_per_block = 49,152` confirms the 49 KB constraint documented in compatibility.md
- FP8 E4M3 tensor creation works — tensor core path should be functional
- CUTLASS is available — can use as reference or fallback
- nvcc missing — DeepGEMM JIT uses NVRTC as alternative, may be sufficient

### Issues found
- PyTorch API: `props.max_shared_memory_per_block` → `props.shared_memory_per_block` (fixed)
- PyTorch API: `props.total_mem` → `props.total_memory` (fixed)

### Next
- Clone upstream DeepGEMM and study kernel code
- Check if NVRTC path works without nvcc
- Identify all tcgen05.fence locations in upstream

## 2026-04-20 — SM120e kernel created (fp8_gemm_nt)

### Kernel port: sm120e_fp8_gemm_1d1d.cuh
- Source: `src/include/deep_gemm/impls/sm120e_fp8_gemm_1d1d.cuh`
- Based on upstream `sm100_fp8_gemm_1d1d.cuh` (563 lines)
- **Delta from SM100** (4 changes only):
  1. L300: `tcgen05_after_thread_sync()` → `__syncthreads()` (MMA warp, before UMMA issue loop)
  2. L324: `tcgen05_after_thread_sync()` → `__syncthreads()` (MMA warp, after SF full barrier wait)
  3. L457: `tcgen05_after_thread_sync()` → `__syncthreads()` (Epilogue warp, after TMEM full barrier)
  4. L529: `tcgen05_before_thread_sync()` → `__syncthreads()` (Epilogue warp, before TMEM empty arrive)
- Function renamed: `sm120e_fp8_gemm_1d1d_impl`
- sm100_utils.cuh reused as-is (MMA/TMA/UMMA/TMEM all work on SM120, only fence is missing)

### Design decisions
- `__syncthreads()` is a stronger barrier than `tcgen05.fence` — correct but may cost perf
- Future optimization: try `__threadfence_block()` or `asm("membar.cta;")` as lighter alternatives
- LAYOUT_AD_M=128 stays hardcoded (UMMA hardware constraint)
- With BLOCK_M=64: UMMA pads to 128 rows, overlapping into B buffer (fits exactly: 16384 ≤ 16384)

### Shared memory budget (BLOCK_M=64, N=64, K=128, stages=1, FP8)
```
Component                  Bytes
CD store (2 stages)       16,384   (64 × 128 × 2)
A buffer (1 stage)         8,192   (64 × 128 × 1)
B buffer (1 stage)         8,192   (64 × 128 × 1)
SF_A (1 stage)               512   (128 × 4)
SF_B (1 stage)               512   (128 × 4)
Barriers (~7 × 64)           448
────────────────────────────────
Total                    ~34,240   << 49,152 limit
```

### Files created
- `src/include/deep_gemm/impls/sm120e_fp8_gemm_1d1d.cuh` — the kernel
- `scripts/patch_deepgemm_sm120e.sh` — patches /opt/deepgemm in Docker
- `tests/test_fp8_gemm_sm120e.py` — correctness test vs torch.matmul

### Outcome
**このアプローチは失敗。** SM120 は tcgen05 命令を持たないことが判明 (下記参照)。

---

## 2026-04-20 — 重大発見: SM120 は SM100 ISA を持たない

### 試したこと

1. **`sm_120a` ターゲットでコンパイル**
   - 結果: `tcgen05.fence` AND `tcgen05.mma` 両方が "not supported on .target sm_120a" で拒否
   - fence だけじゃなく MMA 命令体系そのものが SM120 に存在しない

2. **`sm_100f` (Blackwell family) ターゲットでコンパイル**
   - 結果: ptxas でのコンパイルは成功
   - しかし `cuLibraryLoadFromFile` で cubin をロードすると `cuLibraryGetKernelCount` が 0 を返す
   - SM120 ドライバが SM100 cubin の実行を拒否 → バイナリ非互換

3. **arch_major パッチ** (get_arch_major() で SM120→10 にマッピング)
   - API レベルのチェックは通過するが、JIT コンパイルの段階で上記エラー

### 結論

| 特性 | SM100 (B100/B200) | SM120 (RTX PRO 6000) |
|------|-------------------|---------------------|
| MMA命令 | `tcgen05.mma` (UMMA, TMEM経由) | `mma.sync.aligned.kind::f8f6f4` (レジスタベース) |
| Tensor Memory | あり | **なし** |
| MMAタイルサイズ | M128+xN128+xK32+ | **M16xN8xK32** |
| パイプライン | PipelineAsync (TMEM) | PipelineTmaAsync (SM90同等) |
| データフロー | GMEM→TMA→SMEM→TMEM→UMMA | GMEM→TMA→SMEM→ldmatrix→RMEM→MMA |
| クラスタ | 1-2 CTA multicast | **1x1x1 のみ** (multicast なし) |
| Block Scale SF | UMMA 命令内蔵 | MMA 命令内蔵 (`mxf8f6f4.block_scale`) |

**SM120 は SM100 の「廉価版」ではなく、根本的に異なるアーキテクチャ。**
SM100 カーネルの微修正ポートは不可能。

### CUTLASS SM120 サポートの発見

CUTLASS 4.0 に SM120 専用の完全なサポートが存在:
- `cute/arch/mma_sm120.hpp` — SM120 MMA atoms (16x8x32)
- `cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp` — Block-scaled collective
- `cutlass/gemm/collective/sm120_mma_tma.hpp` — Dense collective
- `cutlass/gemm/collective/builders/sm120_*` — Builder infrastructure
- **`examples/87_blackwell_geforce_gemm_blockwise/`** — SM120 GeForce 向け FP8 GEMM 公式例

### 方針転換

~~SM100 DeepGEMM カーネルのバリア置換ポート~~ → **CUTLASS SM120 ベースで FP8 GEMM を構築**

選択肢:
1. **CUTLASS example 87a をベースに standalone FP8 GEMM** ← 最も確実
2. CUTLASS CollectiveBuilder で DeepGEMM 互換 API ラッパー
3. SM90 系 DeepGEMM カーネルをベースに SM120 MMA に書き換え

**方針: 1 → 2 の順で進める。**

### 参考: CUTLASS example 87a の構成
```
examples/87_blackwell_geforce_gemm_blockwise/
  87a_blackwell_geforce_fp8_bf16_gemm_blockwise.cu
```
- SM120 GeForce 向け FP8 E4M3 x E4M3 → BF16 GEMM
- Blockwise FP32 scale factors
- TileShape: 128x128x128
- ClusterShape: 1x1x1
- `cutlass::arch::Sm120` + `OpClassTensorOp`
- Layout: `cute::tuple<LayoutA, LayoutSFA>` でスケーリング付き

---

## 2026-04-20 — CUTLASS SM120 FP8 GEMM: 初火

### ビルド成功
```bash
# Docker 内で CUTLASS example 87a をそのままビルド
nvcc examples/87_blackwell_geforce_gemm_blockwise/87a_*.cu \
  -o test_87a -std=c++20 -arch=sm_120a \
  -I$CUTLASS/include -I$CUTLASS/tools/util/include -I$CUTLASS/examples/common \
  --expt-relaxed-constexpr --expt-extended-lambda -O3 \
  -DCUTLASS_ARCH_MMA_SM120_SUPPORTED=1
```
- 警告のみ (-diag-suppress=2908 で消せる)
- 正確性テスト: **Passed**

### ベンチマーク結果 (GPU 0, RTX PRO 6000)
```
Size              Runtime     TFLOPS
128x128x128       0.006 ms     0.67
512x512x512       0.008 ms    32.4
1024x1024x1024    0.010 ms   206
2048x2048x2048    0.033 ms   522
4096x4096x4096    0.184 ms   746
4096x4096x8192    0.357 ms   771
```
大サイズで **~770 TFLOPS**。FP8 理論ピーク付近。

### 使用された構成
- MMA: `mma.sync.aligned.kind::f8f6f4.m16n8k32` (SM120 native)
- TileShape: 128x128x128
- ClusterShape: 1x1x1
- Scale factors: Blockwise FP32 (UE8M0)
- Epilogue: BF16 output
- Builder: `cutlass::arch::Sm120` + `OpClassTensorOp` + `cute::tuple<Layout, LayoutSF>`

### 車輪の再発明を避ける
- CUTLASS example 87a がまさに我々が必要なもの
- DeepGEMM のカーネルを一から書くのではなく、CUTLASS の SM120 インフラを活用
- 必要なのは PyTorch/Python ラッパーと DeepGEMM 互換 API

---

## 2026-04-20 — PyTorch C++ Extension 完成

### ビルド
```bash
cd /workspace/DeepGEMM-for-SM120e/src
pip install -e . --break-system-packages
```
- `torch.utils.cpp_extension` でビルド
- CUTLASS 4.0 の SM120 CollectiveBuilder をラップ
- sm_120a ターゲット、C++20

### テスト結果
```
正確性 (FP8 with trivial SF vs BF16 reference):
  128x128x128  diff=0.038  PASS
  256x256x256  diff=0.038  PASS
  512x512x512  diff=0.038  PASS
  1024x1024    diff=0.038  PASS
  2048x2048    diff=0.038  PASS

ベンチマーク (PyTorch extension 経由):
  1024x1024x1024   0.011 ms  191 TFLOPS
  2048x2048x2048   0.033 ms  519 TFLOPS
  4096x4096x4096   0.206 ms  666 TFLOPS
  4096x4096x8192   0.401 ms  686 TFLOPS
```

### Python API
```python
import sm120e_gemm

# Scale factor shapes の取得
sfa_shape, sfb_shape = sm120e_gemm.get_sf_shapes(M, N, K)

# FP8 GEMM 実行
D = sm120e_gemm.fp8_gemm(A_fp8, sf_A, B_fp8, sf_B, alpha=1.0, beta=0.0)
```

### 作成ファイル
- `src/csrc/sm120e_gemm.cu` — CUTLASS wrapper (CUDA kernel + pybind11)
- `src/setup.py` — PyTorch extension build
- `tests/test_sm120e_cutlass.py` — 正確性 + ベンチマーク

---

## 2026-04-20 — vLLM カーネル分析: SM120 のボトルネック特定

### ベースライン
- モデル: Huihui-Qwen3.6-35B-A3B-abliterated-NVFP4 (MoE, 35B params, 3B active)
- コンテナ: `lna-lab/gemma4-inference:latest`
- **ベースライン: 175 tok/s** (GPU 0, single request)

### vLLM のカーネル選択 (SM120)

| レイヤー | カーネル | バックエンド |
|---------|---------|------------|
| Dense Linear (QKV etc) | `FlashInferCutlassNvFp4LinearKernel` | CUTLASS NvFP4 |
| MoE Expert GEMM | `FLASHINFER_CUTLASS` | TensorRT-LLM CUTLASS grouped GEMM |
| Attention | `FLASH_ATTN` | FlashAttention |
| Mamba SSU | Triton | Triton |

### 重大発見: SM120 MoE grouped GEMM の部分的失敗

Autotuner ログで以下のタイル構成が **SM120 で初期化失敗**:
```
FAIL: cutlass_kernel_file_gemm_grouped_sm120_M128_BS_group2.generated.cu
FAIL: cutlass_kernel_file_gemm_grouped_sm120_M256_BS_group0.generated.cu
```

エラー: `Failed to initialize cutlass TMA WS grouped gemm. Error: Error Internal`

これは TensorRT-LLM の CUTLASS SM120 grouped GEMM の一部タイルが正しく動作しないことを示す。
Autotuner は失敗したタイルをスキップして、次善のタイル構成にフォールバックしている。

**推測**: SM120 の共有メモリ制限 (49KB) に起因する可能性が高い。
M128 や M256 のタイルは SM100 (232KB smem) 向けに設計されており、
SM120 では共有メモリが足りず初期化に失敗している。

### 攻めるべきポイント

1. **MoE grouped GEMM が真のボトルネック** — MoE モデルの推論時間の大部分
2. SM120 向けの最適タイルサイズ (M64 or M128 with reduced stages) が使われていない可能性
3. CUTLASS SM120 の grouped GEMM (NVFP4) を正しいタイルで実装すれば改善の余地あり

---

## 2026-04-20 — Lna-Lab SM120e 推論環境の全容解析

### 概要

`lna-lab/gemma4-inference:latest` コンテナに **12 個のパッチ** が仕込まれていた。
vLLM (0.19.1rc1.dev296)、FlashInfer、TensorRT-LLM CUTLASS、PyTorch Inductor に
対してそれぞれ SM120e 対応のハックが施されている。

量子化も同じパッチ環境で実行し、SM120e 最適な NVFP4 形式を生成している。

### 重要な訂正: SM120 の共有メモリ

- **49,152 bytes = per block** (cudaDeviceGetAttribute の値)
- **99 KB = per SM** (CUTLASS/Flash Attn が使うキャパシティ)
- 全パッチは **99 KB** を基準に設計されている

### パッチ一覧

#### 1. Flash Attention Forward/Backward (SM120 サブクラス)
- **場所**: `vllm/vllm_flash_attn/cute/flash_fwd_sm120.py`, `flash_bwd_sm120.py`
- **内容**: SM80 MMA ベースの FlashAttention を SM120 向けにサブクラス化
- **核心**: SM120 は SM80 の `mma.sync.aligned.m16n8k16` を使用 (WGMMA なし)
- **タイル**: head_dim≤64 → 128x128 (48KB), head_dim>64 → 128x64 (64KB)
- **思想**: tcgen05/TMA が無い SM120 では SM80 コードパスが最適

#### 2. `quack/gemm_sm120.py` — カスタム GEMM
- **内容**: warp-level MMA + ldmatrix ベースの GEMM (CUTLASS example `blackwell_geforce/dense_gemm.py` ベース)
- **制約**: FP16/BF16 のみ (FP8 は warp-level MMA 非対応)
- **99 KB smem** 基準で設計

#### 3. FlashInfer CUTLASS Grouped GEMM タイル制限 (核心パッチ)
- **場所**: `flashinfer/jit/gemm/cutlass/generate_kernels.py`
- **内容**: `generate_sm120_grouped_gemm_operations()` が SM120 専用タイルを生成
- **タイル**: [128,128,128], [128,128,256], [256,128,128], [128,256,128]
- **混合 FP8xFP4**: [128,128,128] のみに制限
- **JIT フラグ**: `-DCOMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS`
- **これが MoE grouped GEMM の M128/M256 失敗を根治するパッチ**

#### 4. `CutlassExpertsFp4` に SM120 追加
- **場所**: `vllm/model_executor/layers/fused_moe/cutlass_moe.py`
- **内容**: `is_device_capability_family(120)` を追加

#### 5. `FlashInferExperts` に SM120 追加
- **場所**: `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py`
- **内容**: `is_device_capability(120)` を追加 (SM121 は除外、flashinfer PR #2926 待ち)

#### 6. QUTLASS `matmul_ada_mxf4_bf16_tn`
- **内容**: SM120 最適化 MXFP4 matmul (warp-level MMA + 99KB smem)
- **バイナリ**: `vllm/_C.abi3.so` に `-arch sm_120` でコンパイル済み

#### 7. FlashInfer JIT NVCC フラグ
- **場所**: `flashinfer/jit/core.py`
- **内容**: `sm120a_nvcc_flags`, `sm120f_nvcc_flags`, `sm121a_nvcc_flags` 追加

#### 8. FlashInfer FP4 量子化 JIT
- **場所**: `flashinfer/jit/fp4_quantization.py`
- **内容**: SM120 向け FP4 量子化カーネルの JIT コンパイル

#### 9. FlashInfer TMA FMHA for SM120
- **場所**: `flashinfer/jit/attention/modules.py`
- **内容**: SM120 向け FMHA カーネル (64x128, 64x64 タイル)

#### 10. FlashInfer MLA XQA for SM120
- **場所**: `flashinfer/mla.py`
- **内容**: SM120 → `xqa` バックエンド (SM100 の `trtllm-gen` ではない)

#### 11. PyTorch Inductor バグ修正 (7 個)
- **場所**: `vllm/env_override.py`
- **内容**: piecewise CUDA graph + NVFP4 で必要な Inductor のモンキーパッチ群
  - memory_plan_reuse, graph_partition_signature, should_partition,
  - scheduler, get_raw_stream, constrain_to_fx_strides,
  - fxgraphcache_pickle

#### 12. Marlin W4A8-FP8 SM120
- **場所**: `vllm/model_executor/layers/quantization/utils/marlin_utils.py`
- **内容**: SM120 を Marlin W4A8-FP8 対応に追加

### 設計思想の分析

1. **SM120 = SM80 MMA + SM90 TMA パイプライン + SM100 スケーリング形式**
   - MMA 命令は SM80 互換 (`mma.sync.aligned`)
   - TMA (Tensor Memory Access) は SM90 から継承
   - FP4/FP8 ブロックスケーリングは SM100 形式
   - TMEM/UMMA/tcgen05 は **存在しない**

2. **99 KB が全ての基準** — タイルサイズ、パイプラインステージ数の全てがここから逆算

3. **量子化と推論は一体** — パッチ環境で量子化し、同じパッチ環境で推論

### 次のステップ (根治の道)

**「フォールバックしなくていいように根治する」ために:**

1. これらのパッチを upstream に push できる形で整理
2. CUDA プロファイリングで現在の 175 tok/s のボトルネックを特定
3. 99 KB 制約内でのタイル最適化 (特に grouped GEMM)
4. 我々の CUTLASS SM120 dense FP8 GEMM (686 TFLOPS) を
   この既存のパッチ体系に統合できるか検討

---

## 2026-04-20 — 比較実験 + 公開リポジトリ化

### 比較実験結果

```
nvfp4studio (核心パッチあり, v0.17.1):    173 tok/s
gemma4-inference (全12パッチ, v0.19.1):   175 tok/s
差: +2 tok/s (1.2%)
```

核心パッチ (grouped GEMM タイル制限、QUTLASS SM120 matmul、
CutlassExpertsFp4 SM120 対応) は nvfp4studio に既に含まれていた。
Flash Attn SM120 / quack GEMM の追加効果は ~1%。

### 公開リポジトリ

**https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm**

SM120 NVFP4 推論の方法論を公開リポジトリとして切り出し。

内容:
- SM120 キメラアーキテクチャの技術解析 (docs/sm120-architecture.md)
- 12 パッチの全ドキュメント (patches/)
- ベンチマーク結果
- テスト用モデルリンク (HuggingFace)

### DeepGEMM-for-SM120e プロジェクトの総括

**当初の目的**: DeepSeek の DeepGEMM (SM100 FP8 カーネル) を SM120 にポート
**結果**: SM120 に tcgen05 ISA が存在しないことを実証 → ポート不可能

**副産物**:
1. SM120 の ISA レベルの解剖 (世界初の包括的ドキュメント化)
2. CUTLASS SM120 FP8 dense GEMM: 686 TFLOPS (PyTorch C++ ext)
3. Lna-Lab の SM120 パッチ群 12 個の全容解析と文書化
4. 公開リポジトリ blackwell-geforce-nvfp4-gemm
