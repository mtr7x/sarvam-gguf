#!/usr/bin/env python3
"""
Sarvam-30B → GGUF Conversion Attempt

This script attempts to convert Sarvam-30B from HuggingFace format to GGUF.
Sarvam uses sigmoid routing (not softmax) in its MoE architecture.
The goal is to see how far we get and document what breaks.

Architecture (from HF config):
- 30B total params, 2.4B active
- 19 layers (1 dense + 18 MoE)
- 128 routed experts + 1 shared expert, top-6 routing
- Sigmoid-based routing (the key difference)
- 32 query heads, 4 KV heads, head_dim=128

Run on Google Colab with GPU runtime.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

# === CONFIG ===
MODEL_ID = "sarvamai/sarvam-30b"
WORK_DIR = "/content/sarvam-gguf"
LLAMA_CPP_DIR = "/content/llama.cpp"
MODEL_DIR = f"{WORK_DIR}/sarvam-30b"
OUTPUT_DIR = f"{WORK_DIR}/output"
QUANT_METHODS = ["q4_k_m", "q8_0"]  # try these if conversion works

def log(msg):
    print(f"\n{'='*60}\n[SARVAM-GGUF] {msg}\n{'='*60}")

def run(cmd, check=True):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[-2000:])  # last 2000 chars
    if result.stderr:
        print(f"STDERR: {result.stderr[-2000:]}")
    if check and result.returncode != 0:
        print(f"[FAILED] Return code: {result.returncode}")
        return False
    return True

def step1_setup():
    """Install dependencies and clone llama.cpp"""
    log("Step 1: Setup")
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Install Python deps
    run("pip install -q huggingface_hub hf_transfer sentencepiece protobuf transformers torch numpy gguf", check=False)

    # Clone llama.cpp (latest)
    if not os.path.exists(LLAMA_CPP_DIR):
        run(f"git clone https://github.com/ggml-org/llama.cpp {LLAMA_CPP_DIR}")
    else:
        run(f"cd {LLAMA_CPP_DIR} && git pull")

    # Build llama.cpp (CPU is fine for conversion, GPU for inference)
    run(f"""cd {LLAMA_CPP_DIR} && \
        cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF && \
        cmake --build build --config Release -j$(nproc) --target llama-quantize llama-cli""", check=False)

    return True

def step2_download_model():
    """Download Sarvam-30B from HuggingFace"""
    log("Step 2: Download Sarvam-30B")

    if os.path.exists(f"{MODEL_DIR}/config.json"):
        print("Model already downloaded, skipping...")
        return True

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_DIR,
            ignore_patterns=["*.bin"],  # prefer safetensors
        )
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def step3_inspect_model():
    """Inspect the model config to understand architecture"""
    log("Step 3: Inspect Model Architecture")

    config_path = f"{MODEL_DIR}/config.json"
    if not os.path.exists(config_path):
        print("config.json not found!")
        return False

    with open(config_path) as f:
        config = json.load(f)

    print(json.dumps(config, indent=2))

    # Key fields to check
    model_type = config.get("model_type", "unknown")
    print(f"\nModel type: {model_type}")
    print(f"Architectures: {config.get('architectures', [])}")

    # Check for MoE-specific config
    moe_keys = [k for k in config.keys() if "expert" in k.lower() or "moe" in k.lower() or "router" in k.lower()]
    print(f"MoE-related config keys: {moe_keys}")

    # Check routing type
    if "router_type" in config:
        print(f"Router type: {config['router_type']}")
    if "routing_type" in config:
        print(f"Routing type: {config['routing_type']}")

    # Save config for reference
    with open(f"{OUTPUT_DIR}/model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return config

def step4_check_conversion_support():
    """Check if llama.cpp's convert script recognizes this architecture"""
    log("Step 4: Check convert_hf_to_gguf.py Support")

    convert_script = f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        print("convert_hf_to_gguf.py not found!")
        return False

    # Check what architectures are supported
    result = subprocess.run(
        f"grep -n 'model_type\|class.*Model\|architecture\|sarvam\|sigmoid' {convert_script}",
        shell=True, capture_output=True, text=True
    )
    print("Supported architectures / relevant lines:")
    print(result.stdout[:3000])

    # Also check the gguf-py library for architecture support
    result2 = subprocess.run(
        f"grep -rn 'sarvam\|sigmoid\|SarvamMoe' {LLAMA_CPP_DIR}/gguf-py/ 2>/dev/null || echo 'No sarvam references in gguf-py'",
        shell=True, capture_output=True, text=True
    )
    print("\nSarvam references in gguf-py:")
    print(result2.stdout)

    # Check the models directory
    result3 = subprocess.run(
        f"grep -rn 'sarvam\|sigmoid' {LLAMA_CPP_DIR}/src/ {LLAMA_CPP_DIR}/include/ 2>/dev/null | head -20 || echo 'No sarvam references in src'",
        shell=True, capture_output=True, text=True
    )
    print("\nSarvam references in llama.cpp source:")
    print(result3.stdout)

    return True

def step5_attempt_conversion():
    """Try the actual conversion"""
    log("Step 5: Attempt GGUF Conversion")

    convert_script = f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py"
    output_file = f"{OUTPUT_DIR}/sarvam-30b-f16.gguf"

    print("Attempting: convert_hf_to_gguf.py ...")
    print("This is where we expect it to fail for unsupported architectures.\n")

    result = subprocess.run(
        f"python3 {convert_script} {MODEL_DIR} --outfile {output_file} --outtype f16 2>&1",
        shell=True, capture_output=True, text=True, timeout=1800  # 30 min timeout
    )

    print("STDOUT:")
    print(result.stdout[-5000:])
    print("\nSTDERR:")
    print(result.stderr[-5000:])
    print(f"\nReturn code: {result.returncode}")

    if result.returncode == 0 and os.path.exists(output_file):
        size_gb = os.path.getsize(output_file) / (1024**3)
        print(f"\nSUCCESS! GGUF created: {output_file} ({size_gb:.2f} GB)")
        return output_file
    else:
        print("\nCONVERSION FAILED — documenting the error.")

        # Save the full error for analysis
        with open(f"{OUTPUT_DIR}/conversion_error.txt", "w") as f:
            f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nReturn code: {result.returncode}")

        return False

def step6_quantize(gguf_path):
    """If conversion succeeded, try quantization"""
    log("Step 6: Quantize")

    quantize_bin = f"{LLAMA_CPP_DIR}/build/bin/llama-quantize"
    if not os.path.exists(quantize_bin):
        print("llama-quantize not found, skipping")
        return

    for method in QUANT_METHODS:
        output_file = f"{OUTPUT_DIR}/sarvam-30b-{method}.gguf"
        print(f"\nQuantizing to {method}...")
        result = subprocess.run(
            f"{quantize_bin} {gguf_path} {output_file} {method} 2>&1",
            shell=True, capture_output=True, text=True, timeout=3600
        )
        print(result.stdout[-2000:])
        if result.returncode == 0 and os.path.exists(output_file):
            size_gb = os.path.getsize(output_file) / (1024**3)
            print(f"SUCCESS: {output_file} ({size_gb:.2f} GB)")
        else:
            print(f"FAILED: {method}")
            print(result.stderr[-1000:])

def step7_summary():
    """Print summary of what happened"""
    log("Step 7: Summary")

    print("Files in output directory:")
    for f in sorted(Path(OUTPUT_DIR).glob("*")):
        size = f.stat().st_size
        if size > 1024*1024:
            print(f"  {f.name}: {size/(1024**3):.2f} GB")
        else:
            print(f"  {f.name}: {size} bytes")

def main():
    print("""
    ╔══════════════════════════════════════════╗
    ║  Sarvam-30B → GGUF Conversion Attempt   ║
    ║  github.com/mtr7x | mtrajan.substack    ║
    ╚══════════════════════════════════════════╝
    """)

    step1_setup()
    step2_download_model()
    config = step3_inspect_model()
    step4_check_conversion_support()
    gguf_path = step5_attempt_conversion()

    if gguf_path:
        step6_quantize(gguf_path)
    else:
        print("\n" + "="*60)
        print("CONVERSION FAILED — This confirms the deployment gap.")
        print("llama.cpp does not yet support Sarvam's architecture.")
        print("The sigmoid routing / sarvam_moe model type is not recognized.")
        print("="*60)
        print("\nNext steps:")
        print("1. Check llama.cpp PR #20275 for sarvam_moe support")
        print("2. Patch convert_hf_to_gguf.py with Sarvam architecture")
        print("3. Use vLLM (PR #33942 merged) as reference for tensor mapping")

    step7_summary()

if __name__ == "__main__":
    main()
