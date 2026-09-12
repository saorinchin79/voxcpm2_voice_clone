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

**Studio server (login-gated frontend):**
```sh
python server.py                    # studio.html + /gradio_api proxy on 127.0.0.1:9090
python manage_users.py list         # studio login accounts — see "Studio authentication"
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

## Studio authentication

`server.py` puts a server-side session in front of the studio. `auth.py` holds the stores (PBKDF2-SHA256 password records in `users.json`, sessions in `.sessions.json`, an in-memory per-IP login throttle), `login.html` is the sign-in page, `manage_users.py` is the operator CLI, and `studio.html` carries the user chip and `signOut()`.

`POST /api/login` sets an `HttpOnly` `voxcpm_session` cookie and everything off the public list is checked against it. A browser navigation without a session gets `302 /login.html?next=<original path>`; anything else gets `401 {"error":"Authentication required"}`. The gate covers `studio.html`, the `/gradio_api/*` proxy and `POST /api/i2v`. Reachable without a cookie: `/login.html`, `/login`, `/api/login`, `/api/me`, `/favicon.ico`, any `OPTIONS`, and `/_tmp/*` — `/_tmp/*` is public **deliberately**, because FAL.ai fetches the frame from the internet with no cookie. Static serving is an allowlist (`studio.html`, `login.html`, `favicon.ico`, and the `assets/`, `examples/`, `_tmp/` trees); everything else — `server.py`, `users.json`, `.git`, the logs — is a flat 404 whether signed in or not.

### First boot on a fresh host

`users.json` is gitignored per-host state. It never travels with a `git pull`, so a freshly cloned DP Server has no accounts at all. `seed_defaults()` runs on every start but populates only an *empty* store, so it can never resurrect an account someone deleted.

Non-interactive first start — seeds come from the environment, never from a literal in this public repo:

```sh
VOXCPM_SEED_USERS='admin:<password>,chhay:<password>' python server.py
```

Interactive alternative — start the server normally and create the accounts by hand:

```sh
python manage_users.py add admin    # prompts twice; the password is never read from argv
```

With neither, the first start generates a password per required account. It is printed only when stdout is a terminal — under launchd/systemd stdout is a world-readable log, so instead it lands in a mode-0600 `seed-credentials-<epoch>.txt` at the repo root and the startup line points at it. Sign in, `manage_users.py passwd`, delete that file.

### Day to day

```sh
python manage_users.py list             # role, created, last login
python manage_users.py add <name>
python manage_users.py passwd <name>    # also drops that user's live sessions
python manage_users.py delete <name>    # refuses to remove the last remaining account
```

**No service restart is needed after any of these.** The files, not the in-memory dicts, are the source of truth: both stores `stat()` their file before a read and re-parse only when it moved, and every mutation is a read-modify-write under an `fcntl` lock on a sibling `.lock` file. So a `passwd` typed at the shell really does sign that user out of the already-running server.

### Environment

| Var | Default | Effect |
|---|---|---|
| `VOXCPM_SEED_USERS` | unset | `user:password,user:password` bootstrap seeds, read once at import; a gitignored `.seed_users.json` (`{"user": "password"}`) is consulted next. Applied only while the store is empty. |
| `VOXCPM_FORCE_SECURE_COOKIE` | unset | `1` forces `Secure` on the session cookie regardless of `X-Forwarded-Proto`. |
| `VOXCPM_AUTH_DISABLED` | unset | `1` bypasses the gate entirely — local debugging only; prints a loud warning at startup. |
| `VOXCPM_TRUSTED_PROXIES` | `127.0.0.1 ::1` | Peers whose `X-Forwarded-For` is believed when keying the lockout. Space- or comma-separated. |
| `PUBLIC_HOST` | unset | Origin handed to FAL.ai for `/_tmp/<uuid>.jpg`. Left unset it is derived per request from the forwarded `Host` and scheme, which is what you want behind nginx. |
| `GRADIO_URL` | `http://localhost:$GRADIO_PORT` | Full override for a remote engine — DP Server points it at `https://tts.saorin.me`. |
| `GRADIO_PORT` | `8808` | Only used to build the default `GRADIO_URL`. |
| `HOST` | `127.0.0.1` | Bind address. Keep it on loopback behind nginx. |
| `PORT` | `9090` | Bind port. |

### nginx

The app reads three of these four headers; the proxy block must set all of them.

```nginx
proxy_set_header Host              $host;
proxy_set_header X-Real-IP         $remote_addr;
proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
```

- `X-Forwarded-Proto` decides whether the session cookie carries `Secure`. Pass `$scheme`, never a literal: a `Secure` cookie on a plain-http request is discarded by the browser, so `/api/login` returns 200, nothing is stored, and the next page bounces back to the login form — login silently does nothing, with no error on either side.
- `X-Forwarded-For` is the brute-force lockout key, and is read only when the socket peer is in `VOXCPM_TRUSTED_PROXIES` (loopback by default). `$proxy_add_x_forwarded_for` *appends*, so the value used is the last hop — the one the adjacent proxy wrote itself — and a client-supplied hop 0 cannot rotate past the lockout. Omit the header and every request keys on `127.0.0.1`, where one attacker locks out everybody.
- `Host` is what an incoming `Origin` is compared against before any CORS header is echoed, and is the origin handed to FAL.ai when `PUBLIC_HOST` is unset. Rewriting it breaks both.
- `X-Real-IP` is not read by the app; keep it for nginx's own logs.
- **nginx must proxy everything to `server.py` rather than serving any of it statically** — `server.py` sets the `COOP: same-origin` + `COEP: require-corp` pair that ffmpeg.wasm needs for `SharedArrayBuffer`, and a static location silently drops them and breaks the studio's video export (a bug this deployment has already hit once).
