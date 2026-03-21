# Sarvam-30B → GGUF

Convert [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b) to GGUF by patching llama.cpp with [PR #20275](https://github.com/ggml-org/llama.cpp/pull/20275).

**Pre-built GGUFs available:** [mtrajan/sarvam-30b-GGUF on HuggingFace](https://huggingface.co/mtrajan/sarvam-30b-GGUF)

> **Will this work with Ollama / LM Studio / Jan?**
>
> **Not yet.** These tools bundle mainline llama.cpp, which does not recognize `sarvam_moe`. You will see:
> ```
> error loading model: unknown model architecture: 'sarvam_moe'
> ```
> This GGUF **requires a patched llama.cpp** (with PR #20275 applied) until that PR merges into mainline. Once it does, Ollama / LM Studio / Jan will work automatically on their next update.

## What this does

1. Clones llama.cpp
2. Applies PR #20275 — adds `sarvam_moe` architecture (converter + C++ runtime + tokenizer)
3. Builds `llama-quantize` and `llama-cli`
4. Downloads Sarvam-30B from HuggingFace (~60GB)
5. Converts to GGUF F16
6. Quantizes to Q4_K_M and Q8_0
7. Tests inference with a Hindi prompt

## Pre-built GGUFs

If you just want the files, download from HuggingFace:

| File | Quant | Size | BPW |
|------|-------|------|-----|
| [sarvam-30b-q4_k_m.gguf](https://huggingface.co/mtrajan/sarvam-30b-GGUF) | Q4_K_M | 19 GB | 4.87 |
| [sarvam-30b-f16.gguf](https://huggingface.co/mtrajan/sarvam-30b-GGUF) | F16 | 60 GB | 16.00 |

## Run on Colab (one click)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mtr7x/sarvam-gguf/blob/main/sarvam_gguf_attempt.ipynb)

Runtime → GPU → Run All. Model saves to Google Drive (survives disconnects).

## Run locally / on a server

```bash
git clone https://github.com/mtr7x/sarvam-gguf.git
cd sarvam-gguf
chmod +x patch_and_convert.sh
./patch_and_convert.sh
```

Needs ~120GB disk, cmake, Python 3.10+.

## Why this is needed

Sarvam open-sourced 30B and 105B under Apache 2.0, but no GGUF exists. Mainline llama.cpp doesn't recognize `model_type: "sarvam_moe"` — the converter exits immediately.

Contrary to what you might expect, **sigmoid routing is already supported** in llama.cpp (used by GLM4 and others). The actual blocker is a missing class registration + tensor mappings + C++ graph builder — all provided by PR #20275.

Read the full analysis: [Sarvam. Open is not sovereign](https://mtrajan.substack.com/p/sarvam-open-is-not-sovereign)

## What does NOT work (yet)

| Tool | Status | Why |
|------|--------|-----|
| Ollama | `unknown model architecture` | Waiting on PR #20275 merge |
| LM Studio | `unknown model architecture` | Waiting on PR #20275 merge |
| Jan | `unknown model architecture` | Waiting on PR #20275 merge |
| llama.cpp (mainline) | `unknown model architecture` | PR #20275 not yet merged |
| llama.cpp (patched) | **Works** | This repo builds it |

## What PR #20275 adds (387 lines)

| File | What |
|------|------|
| `convert_hf_to_gguf.py` | `SarvamMoEModel` class — tokenizer conversion (SentencePiece → GPT-2), expert merging, expert bias normalization |
| `gguf-py/gguf/constants.py` | `SARVAM_MOE` arch enum + tensor list (20 tensor types) |
| `src/models/sarvam-moe.cpp` | C++ graph builder — attention with QK norm, RoPE, dense FFN for layer 0, MoE + shared experts for layers 1-18 |
| `src/llama-arch.cpp` | Architecture name + tensor registration |
| `src/llama-model.cpp` | Hyperparameter loading + tensor creation + model info |
| `src/llama-vocab.cpp` | Sarvam-specific tokenizer regex |

## The domino chain

```
PR #20275 merges into llama.cpp          ← we apply this manually
  → GGUF can be created                  ← this repo does it
    → Ollama updates its llama.cpp       ← blocked
      → Unsloth applies dynamic quants   ← blocked
        → ollama run sarvam-30b          ← blocked
```

## Runtime support (March 2026)

| Runtime | Status |
|---------|--------|
| vLLM | ✅ [PR #33942](https://github.com/vllm-project/vllm/pull/33942) merged |
| SGLang | ✅ Works |
| llama.cpp (patched) | ✅ Works — **this repo builds it** |
| llama.cpp (mainline) | ⏳ [PR #20275](https://github.com/ggml-org/llama.cpp/pull/20275) pending |
| Ollama | ❌ Blocked on llama.cpp |
| LM Studio | ❌ Blocked on llama.cpp |
| GGUF | ✅ [mtrajan/sarvam-30b-GGUF](https://huggingface.co/mtrajan/sarvam-30b-GGUF) |

## Architecture

```
sarvamai/sarvam-30b
├── model_type: "sarvam_moe"
├── 30B params, 2.4B active
├── 19 layers (1 dense + 18 MoE)
├── 128 experts + 1 shared, top-6, sigmoid routing
├── 64 query heads, 4 KV heads, head_dim=64
├── vocab_size: 262,144 (Indic-optimized)
└── Apache 2.0
```

## Files

| File | What |
|------|------|
| `patch_and_convert.sh` | Full pipeline: clone, patch, build, download, convert, quantize |
| `sarvam_gguf_attempt.ipynb` | Same pipeline as a Colab notebook with Google Drive storage |
| `convert_sarvam_gguf.py` | Standalone Python script |

## Related

- [Sarvam. Open is not sovereign](https://mtrajan.substack.com/p/sarvam-open-is-not-sovereign)
- [mtrajan/sarvam-30b-GGUF](https://huggingface.co/mtrajan/sarvam-30b-GGUF) — pre-built GGUFs on HuggingFace
- [llama.cpp PR #20275](https://github.com/ggml-org/llama.cpp/pull/20275) — the patch we apply
- [vLLM PR #33942](https://github.com/vllm-project/vllm/pull/33942) — merged
- [Unsloth Dynamic Quantization](https://unsloth.ai/blog/dynamic-4bit) — what Sarvam needs for Indic-safe compression
