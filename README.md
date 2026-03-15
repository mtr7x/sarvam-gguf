# Sarvam-30B → GGUF: Status and Findings

Investigating what it takes to convert [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b) to GGUF for local inference via Ollama, LM Studio, and llama.cpp.

Read: [Sarvam. Open is not sovereign](https://mtrajan.substack.com/p/sarvam-open-is-not-sovereign)

## TL;DR

**Conversion fails immediately.** You don't need to download 60GB to find out — the blocker is a missing ~50-line class registration in llama.cpp, not the sigmoid routing.

## What we found

We downloaded Sarvam-30B's `config.json` and checked it against the latest llama.cpp source:

```
model_type:    "sarvam_moe"           → ❌ No converter class in llama.cpp
architecture:  "SarvamMoEForCausalLM" → ❌ Not registered
```

| Feature | llama.cpp Status |
|---|---|
| Sigmoid routing (`score_function: "sigmoid"`) | ✅ Already supported (auto-detected) |
| MoE expert merging | ✅ Supported (via Qwen2MoeModel) |
| Shared experts | ✅ Supported |
| `moe_intermediate_size` | ✅ Supported |
| **`model_type: "sarvam_moe"`** | ❌ **No class registered** |

The sigmoid routing — which the original article identified as the deployment blocker — is actually **already handled** by llama.cpp. The real blocker is that nobody has registered `SarvamMoEForCausalLM` as a model class in `convert_hf_to_gguf.py` and added the architecture enum + tensor mappings to `gguf-py`.

The fix is structurally simple. Sarvam's architecture is very close to `Qwen2MoeModel`:

```python
@ModelBase.register("SarvamMoEForCausalLM")
class SarvamMoeModel(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.SARVAM_MOE
```

Plus arch enum and tensor name mappings. That's what [llama.cpp PR #20275](https://github.com/ggml-org/llama.cpp/pull/20275) does.

## The domino chain to `ollama run sarvam-30b`

```
llama.cpp PR #20275 merges (converter + C++ runtime)
  → llama.cpp can convert + run sarvam_moe
    → Ollama updates its bundled llama.cpp
      → Unsloth applies dynamic quantization (protects Indic embedding layers)
        → GGUF on HuggingFace
          → ollama run sarvam-30b
```

Every step after the first is routine. Unsloth has done it for DeepSeek R1, Llama 4, Qwen 3, and Gemma 3 — usually within days of release. The bottleneck is the single llama.cpp PR.

## How Unsloth creates GGUFs (and why they can't do it for Sarvam yet)

Unsloth maintains a [fork of llama.cpp](https://github.com/unslothai/llama.cpp) and has a one-liner API:

```python
model.save_pretrained_gguf("dir", tokenizer, quantization_method="q4_k_m")
```

For pre-trained models (like DeepSeek R1), they:
1. Run `convert_hf_to_gguf.py` from their fork
2. Apply **dynamic quantization** — profiling each layer's sensitivity, keeping critical layers (embeddings, early dense layers, `down_proj`) at higher precision while compressing MoE experts to 1.5-2 bits
3. Fix bugs (chat templates, tokenizer, padding tokens)
4. Upload GGUFs to HuggingFace

**For DeepSeek R1**, the architecture was already in mainline llama.cpp. Unsloth's contribution was the quantization strategy. For Sarvam, the architecture itself is missing — so even Unsloth can't start.

The Indic quantization problem from the article is real but comes *after* the architecture registration. Once a GGUF can be created, Unsloth's dynamic quantization approach (selectively preserving embedding and vocabulary layers at higher precision) is exactly what Sarvam needs to protect its Indic language advantage during compression.

## Runtime support (March 2026)

| Runtime | Status | Notes |
|---------|--------|-------|
| vLLM | ✅ Merged | [PR #33942](https://github.com/vllm-project/vllm/pull/33942), Mar 10 |
| SGLang | ✅ Works | Sigmoid routing supported |
| llama.cpp | ⏳ Pending | [PR #20275](https://github.com/ggml-org/llama.cpp/pull/20275) — the single bottleneck |
| Ollama | ❌ Blocked | Needs llama.cpp merge |
| LM Studio | ❌ Blocked | Needs llama.cpp merge |
| GGUF | ❌ None exists | Blocked on llama.cpp |

## Architecture reference

```
sarvamai/sarvam-30b (config.json)
├── model_type: "sarvam_moe"
├── architecture: "SarvamMoEForCausalLM"
├── 30B total params, 2.4B active
├── 19 layers (1 dense + 18 MoE)
├── 128 routed experts + 1 shared, top-6
├── score_function: "sigmoid"
├── hidden_size: 4096, head_dim: 64
├── 64 query heads, 4 KV heads
├── vocab_size: 262144 (large — Indic tokenizer)
├── routed_scaling_factor: 2.5
└── Apache 2.0
```

## Files

- `sarvam_gguf_attempt.ipynb` — Colab notebook that attempts the full conversion pipeline (requires Google Drive for storage)
- `convert_sarvam_gguf.py` — Standalone script version

## Related

- [Sarvam. Open is not sovereign](https://mtrajan.substack.com/p/sarvam-open-is-not-sovereign) — the original analysis
- [Sarvam blog: Open-sourcing 30B and 105B](https://www.sarvam.ai/blogs/sarvam-30b-105b)
- [vLLM PR #33942: Sarvam MoE support](https://github.com/vllm-project/vllm/pull/33942) (merged)
- [llama.cpp PR #20275: sarvam_moe architecture](https://github.com/ggml-org/llama.cpp/pull/20275) (pending)
- [Unsloth Dynamic Quantization](https://unsloth.ai/blog/dynamic-4bit) — the approach Sarvam needs for Indic-safe quantization
- [Unsloth DeepSeek R1 1.58-bit](https://unsloth.ai/blog/deepseekr1-dynamic) — proof that MoE dynamic quantization works
