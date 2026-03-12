# Sarvam-30B → GGUF Conversion Attempt

Attempting to convert [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b) to GGUF for local inference.

## Run it now (one click)

> **Click the badge → set runtime to GPU → Run All**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mtr7x/sarvam-gguf/blob/main/sarvam_gguf_attempt.ipynb)

That's it. The notebook handles everything: clones llama.cpp, downloads Sarvam-30B, attempts conversion, and reports what happens.

## Why this matters

Sarvam open-sourced 30B and 105B models under Apache 2.0. No GGUF exists — Ollama, LM Studio, and llama.cpp can't load them. The blocker: Sarvam's sigmoid routing (vs standard softmax).

Read: [Sarvam. Open is not sovereign](https://mtrajan.substack.com/p/sarvam-open-is-not-sovereign)

## Runtime support status (March 2026)

| Runtime | Status | Notes |
|---------|--------|-------|
| vLLM | ✅ Merged | [PR #33942](https://github.com/vllm-project/vllm/pull/33942), Mar 10 |
| SGLang | ✅ Works | Sigmoid routing supported |
| llama.cpp | ⏳ Pending | [PR #20275](https://github.com/ggml-org/llama.cpp/pull/20275) |
| Ollama | ❌ Blocked | Needs llama.cpp merge first |
| LM Studio | ❌ Blocked | Needs llama.cpp merge first |
| GGUF | ❌ None exists | **This repo attempts it** |

## What the notebook does

1. Installs deps + clones llama.cpp
2. Downloads `sarvamai/sarvam-30b` (~60GB)
3. Inspects model config (architecture, routing type, expert count)
4. Checks if `convert_hf_to_gguf.py` recognizes Sarvam's architecture
5. Attempts F16 GGUF conversion
6. If successful → quantizes to Q4_K_M and Q8_0
7. If failed → documents the exact error and blocker

## Files

- `sarvam_gguf_attempt.ipynb` — Colab notebook (run this)
- `convert_sarvam_gguf.py` — Standalone Python script (same steps, no notebook)

## Architecture reference

```
sarvamai/sarvam-30b
├── 30B total params, 2.4B active
├── 19 layers (1 dense + 18 MoE)
├── 128 routed experts + 1 shared, top-6
├── Sigmoid routing ← the blocker
├── 32 query heads, 4 KV heads
└── Apache 2.0
```

## Related

- [Sarvam blog: Open-sourcing 30B and 105B](https://www.sarvam.ai/blogs/sarvam-30b-105b)
- [vLLM PR #33942: Sarvam MoE support](https://github.com/vllm-project/vllm/pull/33942)
- [llama.cpp PR #20275: sarvam_moe architecture](https://github.com/ggml-org/llama.cpp/pull/20275)
- [Unsloth Dynamic Quantization](https://unsloth.ai/blog/dynamic-4bit) — the approach Sarvam needs
