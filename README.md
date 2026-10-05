# Qwen3.8-Flash-Next on 2× NVIDIA CMP 170HX — Build Recipe & Benchmarks

This repository documents the exact recipe we run in production: which model
checkpoint, which vLLM version with which patches, which (modified) hardware,
and the measured benchmarks of the current build with the full methodology.

API path in our deployment: Caddy/TLS → LiteLLM → vLLM (loopback-only). The
serving stack itself is just Docker + vLLM, described below.

> Last benchmark of the current build: **2026-10-03**. On 2026-10-04 a
> communication-path repair (patch #6 below) was added to the same image; it
> fixes a startup hang and carries no performance claim.

---

## 1. Model

| Item | Value |
|---|---|
| Checkpoint | [`zankich/Qwen3.8-Flash-Next-W4A16-Merlin-INT4PLE-FP8KV`](https://huggingface.co/zankich/Qwen3.8-Flash-Next-W4A16-Merlin-INT4PLE-FP8KV) |
| Revision | `45d5db34e05a826d0ca150d4526bc29fd2958d63` |
| Size | 42 files, 101,361,203,754 bytes (≈94.4 GiB) |
| Quantization | W4A16 (INT4 group size 128, compressed tensors), PLE table packed INT4 + FP16 scales |
| Base model | [`halt95/Qwen3.8-Flash-Next-W4A16-Merlin`](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin) (family: Qwen3.8-Flash-Next). zankich's checkpoint is a repackage of halt95's W4A16 GPTQ work: the 51B-param FP8 PLE table is repacked to INT4 (group-32 symmetric) and FP8-KV scales are recalibrated; everything else carries over unchanged |
| Chat template | [`froggeric/Qwen-Fixed-Chat-Templates`](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) v22.5, revision `855bffc49448e299789730ff92c9b8d834d6cc14` |
| Template file hash | `chat_template.jinja` SHA-256 `e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2` |

Checkpoint config is Qwen4Exp with `ple_layer_ids=[2]`,
`ple_embedding_dtype="int4"`, `split_ngram_parts=128`, `ngram_size=3`,
`heads_per_ngram=8`, `ple_embed_dim=2560`. The checkpoint already ships
`model-pleint4-*.safetensors` + `ple-int4-manifest.json`, so no repacking step
is needed.

Despite the `FP8KV` suffix, KV cache runs **BF16** in this recipe: the Qwen4Exp
QSA path of the pinned vLLM 0.30.0 rejects an FP8 cache; FP8 KV would need a
separate QSA FP8 patch and KV-scale sidecar.

## 2. Hardware

| Item | Value |
|---|---|
| Host | Lenovo ThinkStation P520 |
| GPUs | 2× NVIDIA CMP 170HX, 64 GiB HBM2e each |
| Silicon | GA100 (Ampere datacenter, SM80) — not a GeForce/GA102 derivative |
| SMs per GPU | **74** (unlocked from 70, see below) |
| Driver | NVIDIA 610.43.02, kernel 7.0.0-34-generic (Ubuntu) |
| Power limit | 140 W per GPU, persistent; 200 W observed-draw abort guard |

GPU modifications (all verified live, rollback documented in our private ops
runbooks):

- **VBIOS** `92.00.6D.00.0A` (TPU 268495) flashed on both cards. Max customer
  boost 1695 MHz, 300 W VBIOS ceiling.
- **170tune persistence**: GPC V/F undervolt offsets (+250 / +275), SM clock
  ceiling 1420 MHz (`nvidia-smi -lgc 210,1420`), HBM NDIV 64 (=1728 MHz,
  stock for this VBIOS), relaxed HBM refresh timings (~8.3 µs), re-applied at
  every boot.
- **SM unlock** via [amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)
  (commit `6c442eeb6448b97c803e72b61da344a39e0a26ab`): a driver patch
  (`0016-sm-reconfig.patch`, RECONFIG_PLM + GPC override) raises each card
  from 70 → 74 SMs. Verified by CUDA device attributes and actual `%smid`
  execution across all 74 SMs with 3,145,728 error-free computed values per
  GPU.
- **BAR1 P2P driver patches** (cmpunlocker plus Bayley patches 0011/0013/0015,
  commit `5a7bb4b7e5056306fe49e8b824787659abb19914`) enable NCCL P2P between
  the two cards (`NCCL_P2P_DISABLE=0`, `NCCL_P2P_LEVEL=SYS`).

Host-RAM note: the ~26.8 GiB INT4 PLE table stays in **pinned host memory**
(UVA gather); the device-resident backend is not implemented in the plugin.
Our launcher refuses to start when `MemAvailable` drops below 34 GiB.

## 3. vLLM version and patches

| Item | Pin |
|---|---|
| vLLM | `v0.30.0`, commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| Base image | `vllm/vllm-openai:v0.30.0` amd64, manifest digest `sha256:5f5e535216848d0c52159c8c13a0af04be5f6fe1a84e79914300610796f76d40` |
| INT4-PLE plugin | [`zankich/vllm`](https://github.com/zankich/vllm) branch `v0.29.0z`, commit `6609bfefeaecab938c353dc53d93c2d8045a29e3` (wheel overlay with a source-hash compatibility guard) |
| Final image | `duf/vllm-030-flash-next-int4ple-pp2:0.30.0-mtppp-statefix-speedsuite-baseguard-test-20261003-ea8c0049c1`, ID `sha256:c64a305e9142d6efc7edb5119dc1e25cef46aba8252608abbee4ffc66d575e4a` |
| Statefix base | `...statefix-20261001`, ID `sha256:26c68298be1122b0d13a4ca70c2c6dbc188bc338a4d396ecb64a69541068d509` |

All overlays are hash-guarded file installs onto the pinned base image (each
installer verifies the before-hash and the after-hash). Patch layers, in build
order:

1. **PP2/PLE fix — upstream PR [vllm-project/vllm#56444](https://github.com/vllm-project/vllm/pull/56444)**
   (head `67477bc5a02f9c087c66947049bed285dcaa3bdd`, patch SHA-256
   `5df9a62798ad7da50d8f431ada600575102394e5c939273ced20b45657840c4e`).
   Removes the pipeline-parallel/PLE refusal and transports raw `input_ids`
   with `IntermediateTensors` between PP stages. Without it the model cannot
   run PP2 across two GPUs.
 2. **KV cache spec filter — upstream PR
   [vllm-project/vllm#54793](https://github.com/vllm-project/vllm/pull/54793)**
   (patch SHA-256 `9ce4df3eb3f7d32a0a4ef10f2de5cdb4eaf28c15521b58a3de0ecd9db243d8d7`).
   Filters empty worker-projected `UniformTypeKVCacheSpecs` before KV
   allocation; required for an observed `StopIteration` in
   `vllm/v1/worker/utils.py`.
 3. **MTP draft-head fix for Qwen4Exp** — from
   [vllm-project/vllm#54709](https://github.com/vllm-project/vllm/issues/54709):
   `Qwen4ExpMultiTokenPredictor.forward` checks `intermediate_tensors is None`
   instead of the PP-rank condition. Makes speculative MTP work across PP
   stages (generic MTP-over-PP transport is already in 0.30.0).
 4. **Request-slot state fix — upstream PR
   [vllm-project/vllm#55506](https://github.com/vllm-project/vllm/pull/55506)**
   (open at pin time), pinned on commit
   `5f71d6fac1af38ab89d2d261e6f57d1e88c00090`. Three runtime files rebind the
   Mamba tables/indexing to persistent request slots. This fixed silent
   state-corruption failures we traced through MTP EOS anomalies.
 5. **SpeedSuite** — five upstream speed PRs applied as a verified file overlay
   on the statefix image:

   | PR | Title | State at pin |
   |---|---|---|
   | [#58762](https://github.com/vllm-project/vllm/pull/58762) | [Perf][MRV2] Reuse Mamba/GDN metadata across KV cache groups | merged |
   | [#58763](https://github.com/vllm-project/vllm/pull/58763) | [Perf][GDN] Slice pure spec-decode rows instead of a host-mask gather | merged |
   | [#58449](https://github.com/vllm-project/vllm/pull/58449) | [Spec Decode][Qwen3.8] Fused draft metadata updates for the QSA side cache | **open** |
   | [#58114](https://github.com/vllm-project/vllm/pull/58114) | [Perf][Qwen3.8] Reduce PLE metadata construction overhead | merged |
   | [#57097](https://github.com/vllm-project/vllm/pull/57097) | [Qwen3.8-Flash-Next] Fuse main QK-norm/RoPE/gate and KV-cache write into the QSA pre-indexer launch | merged |

   Re-apply each PR onto a v0.30.0 checkout; some hunks need 3-way merges
   because of context-only upstream drift (details and every before/after file
   hash are in our private `speedsuite/manifest.json`). Re-check #58449
   against upstream before reuse — it was still open at pin time.
 6. **Batched P2P communication fix (site-local, 2026-10-04)** — regular
   cold starts occasionally stalled after weight loading while PyTorch
   lazily created an extra pair communicator for unbatched P2P
   ([ProcessGroupNCCL](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3883-L3926)).
   A source-hash-checked script rewrites exactly three GPU
   tensor-dictionary `isend`/`irecv` call sites in `parallel_state.py` to
   `torch.distributed.batch_isend_irecv`, and the result is mounted
   read-only. Source SHA-256
   `d9b10ad56833ebc08a753e1a050ef0989c30d4134021b7b7b00150b7bd01adc6`,
   patched SHA-256
   `421603b499aa2abf897ee78b4b6c597628a630e98b97139a469e64eeb74f79ec`.
   The next regular morning start succeeded on the first attempt.

Driver-side patches (SM unlock, BAR1 P2P) are listed under Hardware; they are
outside the container but required for the P2P behavior above.

## 4. Selected serve profile (`p-prefill2048-mtp3-speedsuite`)

Effective serving configuration (all bound to loopback only):

| Setting | Value |
|---|---|
| Parallelism | PP2 / TP1 (`--pipeline-parallel-size 2 --tensor-parallel-size 1 --distributed-executor-backend mp`) |
| Speculative decoding | MTP, `{"method":"mtp","num_speculative_tokens":3}` |
| Max model length | 262144 |
| Max sequences | 8 |
| Max batched tokens | 2048 (chunked prefill on) |
| CUDA graphs | PIECEWISE, capture token sizes `[1,2,4,8,16,32]` |
| KV cache | BF16 |
| GPU memory utilization | 0.90 |
| NCCL | P2P enabled (`NCCL_P2P_DISABLE=0`, `NCCL_P2P_LEVEL=SYS`) + batched-P2P override (#6) |
| Env | `VLLM_COMPUTE_NANS_IN_LOGITS=1` (diagnostic) |
| Parsers | `--reasoning-parser qwen3 --tool-call-parser qwen3_xml`, froggeric template mounted read-only |
| Multimodal | up to 4 images and 1 video (2 frames) per prompt |

Depth 3 on a one-MTP-layer checkpoint is intentional and valid: the checkpoint
declares `mtp_num_hidden_layers: 1` and vLLM reuses that layer via
`spec_step_idx % num_mtp_layers`; `SpeculativeConfig` only rejects a `k` that
is not a multiple of `n_predict`. The accept rate is therefore **measured**,
not assumed (52.27 % in the benchmark below).

Caveat from hands-on experience: `--max-num-seqs 8` with PP2 does not create
eight independent in-flight forwards per rank — the V1 runner caps concurrent
batches at the pipeline size.

## 5. Benchmarks — current build (2026-10-03, 74 SMs)

### Methodology

- Harness: our `llama-benchy` runner driving the OpenAI-compatible endpoint
  directly on loopback (no gateway in the measurement path).
- Binding matrix: `--pp 16384 --tg 2048 --concurrency 1 4 --runs 3 --exact-tg
  --no-cache --no-warmup --skip-coherence --latency-mode none
  --no-adapt-prompt`.
- Warmth control: a separate full-shape C1+C4 warmup at the same load shape
  before the measured matrix; warmup runs are excluded from scoring.
- Isolation audit on the scored window: exactly 15 successful requests, all
  ending with `length`, 30,720 output tokens (15 × 2048), zero prefix-cache
  hits, zero preemptions, zero errors; queues empty before/after. Metric
  snapshots taken before/after to prove no foreign traffic.
- Telemetry every 2 s: GPU 0 max 159.66 W / 55 °C, GPU 1 max 162.77 W / 56 °C;
  limits held at 140 W, both peaks under the 200 W abort threshold. No
  Xid/OOM/EngineDeadError.
- Statistics: mean ± population SD over the three scored runs per cell.

### Results (profile `p-prefill2048-mtp3-speedsuite`, 74 SMs, NCCL-P2P, 140 W)

| Metric | Mean ± SD (tok/s) | Min–Max |
|---|---:|---:|
| Prefill C1 (16k prompt) | 6213.66 ± 99.27 | 6095.06–6338.01 |
| Decode C1 (2k out) | 133.39 ± 22.81 | 108.10–163.37 |
| Prefill C4 | 6639.25 ± 34.79 | 6595.91–6681.10 |
| Decode C4, aggregate of 4 requests | 245.17 ± 1.17 | 243.53–246.18 |

MTP depth 3 confirmed active; draft accept rate 52.27 % over the whole
benchmark. Decode C1 has notable run-to-run dispersion (see SD) — we report it
rather than the best run.

### Measurement caveats we stand behind

- The C1-Decode 70-vs-74 SM delta (+16 %) and C4 delta (−2.5 %) are **not** a
  controlled SM on/off A/B test; the SM unlock was a driver-level update
  followed by a fresh warm benchmark.
- A `FULL_AND_PIECEWISE` CUDA-graph trial on this runtime produced Xid 13/43
  after the matrix and was rolled back; PIECEWISE is the qualified mode.
- The 2026-10-04 batched-P2P repair matrix passed functionally but ran with
  production traffic in the window; its numbers are deliberately **not**
  published as a speed comparison.

## 6. Reproducing this

1. Patch/build: start from `vllm/vllm-openai:v0.30.0` at the pinned digest and
   apply the six layers in section 3 in order, verifying every documented
   before/after SHA-256. Every overlay fails closed on hash mismatch.
2. Download the model snapshot and the pinned chat template; verify file
   counts/sizes/hashes against the pins in section 1.
 3. Serve with the profile in section 4 on a PP2-capable two-GPU host; note the
    pinned-host-memory requirement for the INT4 PLE table.
4. The driver-side SM unlock and BAR1 P2P patches are optional for
   functionality (the model runs with 70 SMs and `NCCL_P2P_DISABLE=1`) but
   required to match the published throughput. Modifying VBIOS/drivers is at
   your own risk.

Licensing: this README text is free to reuse. All patches/models belong to
their respective upstream authors and carry their original licenses; consult
each upstream repo/checkpoint page before commercial use.
