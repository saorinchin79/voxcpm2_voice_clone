# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (development):**
```sh
pip install -e ".[dev]"
# or with uv:
uv pip install -e ".[dev]"
```

**Run tests:**
```sh
pytest tests/
# Single test file:
pytest tests/test_cli.py
# Single test:
pytest tests/test_cli.py::test_parser_defaults_to_voxcpm2
```

**Lint / format:**
```sh
flake8 src/
black src/
```

**Web demos:**
```sh
python app.py --port 8808           # Gradio inference demo
python lora_ft_webui.py             # LoRA fine-tuning WebUI (port 7860)
```

**Fine-tuning:**
```sh
python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml
python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v2/voxcpm_finetune_all.yaml
```

**Validate training manifest before fine-tuning:**
```sh
voxcpm validate --manifest path/to/manifest.jsonl
```

## Architecture

VoxCPM is a **tokenizer-free, diffusion autoregressive** TTS system. It works entirely in the latent space of AudioVAE and follows a four-stage pipeline:

```
Text → LocEnc → TSLM (MiniCPM4 LM) → RALM → LocDiT → AudioVAE decode → Waveform
```

### Public API entry point

`src/voxcpm/core.py` — `VoxCPM` class. `from_pretrained()` downloads the model (HF Hub or ModelScope), reads `config.json` to detect `architecture` (`"voxcpm"` vs `"voxcpm2"`), and dispatches to `VoxCPMModel` or `VoxCPM2Model`. The `generate()` / `generate_streaming()` methods on `VoxCPM` wrap the underlying model and optionally apply the ZipEnhancer denoiser post-processing.

### Model implementations

- `src/voxcpm/model/voxcpm.py` — `VoxCPMModel`: V1 / V1.5 model (0.5B–0.8B). Also defines `LoRAConfig` and `LoRAInfo` used by both versions.
- `src/voxcpm/model/voxcpm2.py` — `VoxCPM2Model`: V2 model (2B, 30 languages, 48kHz). Shares the same module building blocks but with the V2 AudioVAE, updated LocDiT, and a larger MiniCPM4 backbone.

### Neural modules (`src/voxcpm/modules/`)

| Module | Purpose |
|---|---|
| `audiovae/audio_vae.py` | AudioVAE V1 — encode/decode 16kHz waveforms to/from latent space |
| `audiovae/audio_vae_v2.py` | AudioVAE V2 — asymmetric encoder (16kHz in) / decoder (48kHz out) used by VoxCPM2 |
| `locenc/local_encoder.py` | LocEnc — local acoustic encoder that compresses audio into 6.25Hz feature tokens |
| `minicpm4/model.py` | MiniCPM4 backbone transformer (language model stage) |
| `locdit/local_dit.py` | LocDiT V1 — diffusion transformer decoder |
| `locdit/local_dit_v2.py` | LocDiT V2 — updated diffusion transformer used by VoxCPM2 |
| `locdit/unified_cfm.py` | Flow Matching wrapper (`UnifiedCFM`) around LocDiT |
| `layers/lora.py` | LoRA injection utilities (`apply_lora_to_named_linear_modules`) |
| `layers/scalar_quantization_layer.py` | Scalar quantization used in the latent pipeline |

### CLI (`src/voxcpm/cli.py`)

Subcommand-based CLI: `voxcpm design`, `voxcpm clone`, `voxcpm batch`, `voxcpm validate`. `_build_parser()` returns the argparse parser (used directly in tests without loading the model). The CLI defers `soundfile` and `voxcpm.core` imports until inference commands actually run, so `--help` and `validate` are fast.

### Training (`src/voxcpm/training/`)

- `data.py` — `load_audio_text_datasets()` loads JSONL manifests via HF `datasets`, casts audio columns to the target sample rate
- `validate.py` — pre-flight manifest validation (format, missing files, audio integrity)
- `config.py` — training hyperparameter config (argbind-based)
- `accelerator.py` / `state.py` / `tracker.py` — Accelerate-based training loop helpers
- `packers.py` — `AudioFeatureProcessingPacker` batches and packs audio features for efficient training

### Config files

YAML configs in `conf/` are consumed by `train_voxcpm_finetune.py` via argbind. `conf/voxcpm_v2/` holds VoxCPM2-specific configs. Top-level `conf/voxcpm_finetune_lora.yaml` and `conf/voxcpm_finetune_all.yaml` are aliases that delegate to the versioned configs.

### Denoiser (`src/voxcpm/zipenhancer.py`)

Optional post-processing stage. Wraps ModelScope's `speech_zipenhancer_ans_multiloss_16k_base`. Disabled by default in production (`load_denoiser=False`). Enabled when `VoxCPM` is instantiated with `enable_denoiser=True` and a valid `zipenhancer_model_path`.

## Key design notes

- The `architecture` field in `config.json` inside the model directory drives dispatch between V1 and V2. Don't rely on the HF model ID alone.
- LoRA is injected into named linear modules at construction time (`apply_lora_to_named_linear_modules`). When loading a checkpoint that has mismatched LoRA rank, the loader logs skipped keys rather than raising — watch for those warnings.
- `torch.compile` is applied at startup (`optimize=True` default). Disable for debugging or profiling by passing `optimize=False` to the constructor.
- Training manifests must be JSONL with `audio`, `text`, and optionally `ref_audio` columns. Run `voxcpm validate --manifest` before any fine-tuning run to catch issues early.
