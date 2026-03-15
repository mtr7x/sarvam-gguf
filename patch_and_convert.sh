#!/bin/bash
#
# Sarvam-30B → GGUF Conversion
#
# This script:
# 1. Clones llama.cpp
# 2. Applies PR #20275 (sarvam_moe architecture support)
# 3. Builds the converter + quantizer
# 4. Downloads Sarvam-30B
# 5. Converts to GGUF (F16)
# 6. Quantizes to Q4_K_M and Q8_0
#
# Requirements:
#   - ~120GB disk (60GB model + 60GB GGUF + quantized)
#   - Python 3.10+
#   - cmake, build-essential
#   - pip: huggingface_hub, hf_transfer, sentencepiece, protobuf, numpy, torch, gguf
#
# Usage:
#   chmod +x patch_and_convert.sh
#   ./patch_and_convert.sh
#
# On Google Colab, use the notebook instead — it mounts Google Drive for storage.

set -euo pipefail

WORK_DIR="${WORK_DIR:-$(pwd)}"
LLAMA_CPP_DIR="${WORK_DIR}/llama.cpp"
MODEL_DIR="${WORK_DIR}/sarvam-30b"
OUTPUT_DIR="${WORK_DIR}/output"
PR_NUMBER=20275

echo "╔══════════════════════════════════════════╗"
echo "║  Sarvam-30B → GGUF Conversion            ║"
echo "║  Applies llama.cpp PR #${PR_NUMBER}              ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# --- Step 1: Clone llama.cpp ---
echo "=== Step 1: Clone llama.cpp ==="
if [ ! -d "${LLAMA_CPP_DIR}" ]; then
    git clone --depth 50 https://github.com/ggml-org/llama.cpp "${LLAMA_CPP_DIR}"
else
    echo "llama.cpp already cloned, pulling latest..."
    cd "${LLAMA_CPP_DIR}" && git pull && cd "${WORK_DIR}"
fi

# --- Step 2: Apply PR #20275 ---
echo ""
echo "=== Step 2: Apply PR #${PR_NUMBER} (sarvam_moe support) ==="
cd "${LLAMA_CPP_DIR}"

# Check if already applied
if grep -q "SarvamMoEForCausalLM" convert_hf_to_gguf.py 2>/dev/null; then
    echo "PR already applied, skipping."
else
    echo "Fetching and applying PR #${PR_NUMBER}..."
    gh pr checkout ${PR_NUMBER} --force 2>/dev/null || {
        # Fallback: fetch the PR diff and apply it
        echo "gh not available or PR checkout failed, trying patch..."
        gh pr diff ${PR_NUMBER} > /tmp/sarvam_pr.patch 2>/dev/null || {
            echo "Fetching patch via curl..."
            curl -sL "https://github.com/ggml-org/llama.cpp/pull/${PR_NUMBER}.patch" > /tmp/sarvam_pr.patch
        }
        git apply /tmp/sarvam_pr.patch || {
            echo "Clean apply failed, trying with 3-way merge..."
            git apply --3way /tmp/sarvam_pr.patch || {
                echo "❌ Could not apply patch. You may need to resolve conflicts manually."
                exit 1
            }
        }
    }
    echo "✅ PR applied"
fi

# Verify
if grep -q "SarvamMoEForCausalLM" convert_hf_to_gguf.py; then
    echo "✅ SarvamMoEForCausalLM registered in converter"
else
    echo "❌ SarvamMoEForCausalLM NOT found — patch may have failed"
    exit 1
fi

cd "${WORK_DIR}"

# --- Step 3: Build llama.cpp ---
echo ""
echo "=== Step 3: Build llama.cpp ==="
cd "${LLAMA_CPP_DIR}"

# Detect CUDA
if command -v nvcc &> /dev/null; then
    echo "CUDA detected, building with GPU support..."
    CUDA_FLAG="-DGGML_CUDA=ON"
else
    echo "No CUDA, building CPU-only..."
    CUDA_FLAG="-DGGML_CUDA=OFF"
fi

cmake -B build \
    -DBUILD_SHARED_LIBS=OFF \
    ${CUDA_FLAG} \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc) \
    --target llama-quantize llama-cli

# Verify build
for tool in llama-quantize llama-cli; do
    if [ -f "build/bin/${tool}" ]; then
        echo "✅ ${tool} built"
    else
        echo "❌ ${tool} NOT found"
        exit 1
    fi
done

cd "${WORK_DIR}"

# --- Step 4: Install Python deps ---
echo ""
echo "=== Step 4: Install Python dependencies ==="
pip install -q huggingface_hub hf_transfer sentencepiece protobuf numpy torch gguf transformers

# --- Step 5: Download model ---
echo ""
echo "=== Step 5: Download sarvamai/sarvam-30b ==="
mkdir -p "${MODEL_DIR}" "${OUTPUT_DIR}"

if [ -f "${MODEL_DIR}/config.json" ]; then
    echo "Model already downloaded, skipping."
else
    echo "Downloading (~60GB, may take 15-30 min)..."
    HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='sarvamai/sarvam-30b',
    local_dir='${MODEL_DIR}',
    ignore_patterns=['*.bin', '*.h5', '*.msgpack', 'original/**'],
    resume_download=True,
)
print('✅ Download complete')
"
fi

echo "Model size:"
du -sh "${MODEL_DIR}"

# --- Step 6: Convert to GGUF (F16) ---
echo ""
echo "=== Step 6: Convert to GGUF (F16) ==="
GGUF_F16="${OUTPUT_DIR}/sarvam-30b-f16.gguf"

if [ -f "${GGUF_F16}" ]; then
    echo "F16 GGUF already exists, skipping conversion."
else
    python3 "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
        "${MODEL_DIR}" \
        --outfile "${GGUF_F16}" \
        --outtype f16

    if [ -f "${GGUF_F16}" ]; then
        SIZE=$(du -sh "${GGUF_F16}" | cut -f1)
        echo "✅ F16 GGUF created: ${GGUF_F16} (${SIZE})"
    else
        echo "❌ Conversion failed"
        exit 1
    fi
fi

# --- Step 7: Quantize ---
echo ""
echo "=== Step 7: Quantize ==="

QUANTIZE="${LLAMA_CPP_DIR}/build/bin/llama-quantize"

for METHOD in q8_0 q4_k_m; do
    GGUF_QUANT="${OUTPUT_DIR}/sarvam-30b-${METHOD}.gguf"
    if [ -f "${GGUF_QUANT}" ]; then
        echo "${METHOD} already exists, skipping."
    else
        echo "Quantizing to ${METHOD}..."
        "${QUANTIZE}" "${GGUF_F16}" "${GGUF_QUANT}" "${METHOD}"
        if [ -f "${GGUF_QUANT}" ]; then
            SIZE=$(du -sh "${GGUF_QUANT}" | cut -f1)
            echo "✅ ${METHOD}: ${GGUF_QUANT} (${SIZE})"
        else
            echo "❌ ${METHOD} failed"
        fi
    fi
done

# --- Summary ---
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Done!                                   ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "Output files:"
ls -lh "${OUTPUT_DIR}"/*.gguf 2>/dev/null || echo "No GGUF files found"
echo ""
echo "To run with llama.cpp:"
echo "  ${LLAMA_CPP_DIR}/build/bin/llama-cli \\"
echo "    --model ${OUTPUT_DIR}/sarvam-30b-q4_k_m.gguf \\"
echo "    --prompt 'நமஸ்காரம்! Tell me about India.' \\"
echo "    --n-gpu-layers 99"
echo ""
echo "To run with Ollama (after Ollama updates its llama.cpp):"
echo "  ollama create sarvam-30b -f Modelfile"
echo "  ollama run sarvam-30b"
