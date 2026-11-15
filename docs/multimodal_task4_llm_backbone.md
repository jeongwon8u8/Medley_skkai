# Task 4 — LLM backbone, PEFT modes and hardware preflight

Implemented 2026-11-15 (Asia/Seoul), on `feat/multimodal-framework` after local Task 3 commit `6734a9fde744447a5a28a5ca18ab62bd9e3d5d0d`. This task provides causal-LM loading/preparation, hidden-state access, PEFT adaptation and explicit preflight. It implements no MoCA model, clinical serializer, EEG module, fusion, optimizer, task loss or training/CV loop. Task 5 is not started. Final-review authorization covers one Task 4 commit and a development-branch SSH push attempt; the actual Git outcome is reported separately. No earlier commits are rewritten and no merge/PR is requested.

## Files and integration boundary

Added:

- `medley_pd/multimodal/llm_backbone.py`
- `medley_pd/multimodal/llm_preflight.py`
- `medley_pd/tests/test_multimodal_llm_backbone.py`
- `medley_pd/tests/test_multimodal_llm_preflight.py`
- `medley_pd/requirements-multimodal.txt`
- `medley_pd/requirements-multimodal-gpu.txt`
- This document.

Modified only `medley_pd/multimodal/config.py` to add backward-compatible optional LLM runtime settings. Existing four JSON configurations, existing Medley config loader, EEG preprocessing/features, adapter, Dataset/Collator, fixed splits, baseline/evaluation implementations and existing tests are preserved. Task 2/3 still supply clinical dictionaries/EEG batches; this task does not consume patient data or turn those dictionaries into text.

## Backbone APIs

| API | Contract |
|---|---|
| `load_llm_backbone(config, *, allow_download=False)` | Explicit HF AutoConfig/AutoTokenizer/AutoModelForCausalLM loading; returns LLMBackbone. Requires concrete model_id. Local-only by default |
| `prepare_llm_backbone(config, model, tokenizer)` | Configure an explicitly supplied unadapted local causal model; no model creation/download. Supports tiny synthetic models. Changes the supplied model/tokenizer in place; copies resolved configuration |
| `backbone.model`, `.tokenizer`, `.mode` | Prepared HF/PEFT objects and configured adaptation mode |
| `backbone.embedding_layer`, `.hidden_dim` | Actual token embedding module and width H; width is not guessed from model branding |
| `backbone.forward_hidden(input_ids=..., inputs_embeds=..., attention_mask=...)` | Final layer `[B,L,H]`; exactly one input representation; explicit binary mask required |
| `select_prediction_state(hidden, attention_mask, positions=None)` | `[B,H]` at each row's last real position, or explicit int64 `[B]` positions; must select unmasked positions |
| `resolve_lora_targets(model, requested)` | Actual supported Linear/Conv1D modules, not a fixed family-independent q_proj assumption |
| `parameter_statistics(model)` | Total/trainable count and percentage, base freezing and trainable LoRA counts. Uses HF/PEFT count APIs where available, including their quantized-weight accounting |
| `backbone.metadata()` | Model type/ID, requested/resolved revision, embedding dtype/device, hidden size, pad settings, chat-template presence, actual LoRA targets and parameter statistics |
| `backbone.resolved_config` | Effective Task 1-compatible snapshot: actual padding side, device, selected target modules and model commit hash as revision when available |
| `probe_backbone(backbone)` | Explicit tiny synthetic forward/backward check. Clears gradients first; intended for a freshly prepared model, not mid-training |

`BackboneError` describes unsupported inputs, model policies or runtime prerequisites. No automatic mode/precision fallback or CPU/GPU offloading is performed. Caller batches must be on the embedding device. Explicit single-device placement avoids using an inference-only auto-offload plan as a training plan.

### Hidden states, masks and position selection

`input_ids` must be int64 vocabulary indices `[B,L]`. Alternatively, `inputs_embeds` must be finite floating `[B,L,H]` on the embedding device, with a dtype suitable for the model/autocast policy. Inputs remain connected to autograd. Nonempty shapes, vocabulary bounds, hidden width and configured/model context budget are checked. No truncation occurs. The model is called with output_hidden_states=True, return_dict=True and use_cache=False; the final returned layer is selected.

Attention masks are binary `[B,L]`, with at least one real position per row. Position IDs are derived using cumulative real-token counts so left- and right-padding place actual tokens at 0..N-1. Selection uses the maximum **physical unmasked index**, not L-1 or sum(mask)-1; tests also cover internal mask holes. Explicit prediction positions are bounds-checked and cannot point into padding. No prediction suffix or special prediction token is defined.

Final hidden states must also be finite. The wrapper rejects NaN/Inf model outputs before a numerical failure can be reported as a usable forward pass.

This is an **LLM token attention mask**. Task 3's eeg_mask describes EEG sequence existence, and its artifact_flags describe potential EEG artifacts. No EEG mask conversion or combined EEG/text attention mask is implemented here.

## Adaptation modes

### Frozen

All pretrained parameters are frozen and no LoRA adapters are attached. Already PEFT-wrapped models are rejected even in Frozen mode. Forward does not use no_grad or detach outputs: a trainable external embedding/projector input can receive gradients through frozen operations. Models start in eval mode; callers control later train/eval state. The initial config permits either full precision (`quantization=none`) or frozen NF4; NF4 still requires the quantized backend. Selecting Frozen does not silently change the existing quantization setting.

### LoRA

Full-precision loading follows compute_dtype, base parameters are frozen, and PEFT attaches only trainable LoRA weights with configured rank/alpha/dropout and bias=none. Null target_modules means enumerate actual supported Linear/Conv1D modules while excluding the LM output head. Explicit target names match exact qualified names or suffixes; unmatched targets fail. This supports GPT2 Conv1D projections and Qwen/Gemma Linear projections without assuming identical names. The resolved full target list is recorded for snapshots.

### QLoRA

The loading factory requests real HF BitsAndBytesConfig: load_in_4bit=True, NF4, double quantization and the configured compute dtype. It loads on an explicit CUDA device, checks bitsandbytes/native-library prerequisites and uses PEFT prepare_model_for_kbit_training before attaching LoRA. The prepared model must report NF4 4-bit loading and contain actual Linear4bit weights with a quantization state. Full-precision LoRA or attribute-only claims cannot be silently labeled QLoRA. Missing CUDA/bitsandbytes, unavailable native backend or mismatched quantization fail explicitly.

Non-reentrant checkpointing is requested when enabled. New Transformers activation-checkpointing APIs are used when available; the legacy interface remains a compatibility fallback. PEFT's k-bit preparation uses its installed supported API. Some PEFT internals still emit Transformers deprecation warnings; the tested stack is pinned rather than assuming Transformers 6 compatibility.

**Actual NF4 model loading/forward/backward was not executed in Task 4.** Backend/config construction and failure behavior are tested separately from real GPU execution. No quantized run is claimed from a mocked BitsAndBytesConfig check.

## Model families and access boundaries

CPU synthetic tests instantiate tiny **GPT2, Qwen2 and Gemma** causal architectures from local configurations with random weights. They verify interfaces, not pretrained medical performance or every version in those families. No production HF model ID is selected; existing model_id/revision defaults remain null. The factory requires a concrete ID/path before loading and never invents one.

Inspection uses actual model configuration. Multimodal vision/audio/text-composite configs and encoder-decoder models fail with an explicit text-only decoder requirement before weights are loaded. Unsupported AutoModelForCausalLM configurations fail rather than being forced into the wrong loader. Custom remote code is disabled by default; enabling it requires explicit trust_remote_code=True, a reason and a revision.

MedGemma is **not validated or generically supported** by the text-only adapter. In particular, Google's [medgemma-4b-it model card](https://huggingface.co/google/medgemma-4b-it) identifies an image-text-to-text Gemma3 model; that composite architecture needs a deliberately implemented adapter. Do not interpret the successful tiny Gemma text test as MedGemma compatibility. Gated access, licenses, authentication, exact tokenizer/chat templates and real-model revision compatibility require verification for whichever model is eventually selected. No tokens are accepted as config fields or printed by preflight; HF's normal external authentication facilities apply at a later explicit load.

## Tokenizer and padding policy

Existing tokenizer special tokens are retained. A valid existing pad token is used. If absent, default policy `require` fails; explicit `pad_token_policy=eos` reuses the existing EOS token. It adds no new token, changes no vocabulary size and performs no automatic embedding resize. Vocabulary IDs must fit the model's embedding table. When EOS is padding, an explicit attention mask is essential: token ID equality cannot distinguish actual EOS from padding. Chat-template presence is reported, not applied.

Null padding_side inherits the tokenizer's native left/right policy; an explicit override is respected and the effective side is captured. The model pad_token_id is aligned with the selected tokenizer. These operations do not implement clinical serialization, a prediction suffix or clinical text tokenization rules. Synthetic tokenizer tests use only generic words such as "hello synthetic text".

## Configuration extension and snapshots

Existing required LLM settings retain their meanings: model_id/revision, Frozen/LoRA/QLoRA tuning, quantization, compute_dtype, LoRA rank/alpha/dropout and max_text_tokens. Optional Task 4 fields are injected on loading older configs/snapshots before strict unknown-key validation, so all four existing JSON files remain unchanged and valid:

| Field | Default / validation |
|---|---|
| cache_dir | null or nonempty string; HF cache policy applies when null |
| local_files_only | true, Boolean; no network by default |
| trust_remote_code | false, Boolean |
| remote_code_reason | null or nonempty string; required together with revision when remote code is explicitly enabled |
| target_modules | null for architecture inspection, or distinct nonempty module-name strings |
| max_sequence_tokens | 512, positive integer, at least max_text_tokens; includes all eventual token positions |
| padding_side | null / left / right |
| pad_token_policy | require / eos |
| device | auto / cpu / cuda[:index] |

`save_resolved_config(backbone.resolved_config, output_path)` retains effective settings through the existing exclusive-create snapshot API. If HF supplies a resolved commit hash it replaces an ambiguous requested revision in that effective snapshot; metadata retains the requested revision separately. Local synthetic models have no remote commit hash. Snapshots do not package weights, tokenizer files or authorization and do not by themselves guarantee bitwise reproducibility.

## Dependencies and exact environment changes

The existing root pyproject and legacy Medley requirements were inspected and left unchanged. Tests reused `/tmp/medley-task1-tests`, a venv with system-site-packages. The install plan was reported and checked using pip dry-run while constraining the existing torch/numpy versions. Then the following was executed in that venv only:

```bash
/tmp/medley-task1-tests/bin/python -m pip install 'transformers==5.23.0' 'peft==0.21.3' 'accelerate==1.15.0' 'safetensors==0.8.0' 'torch==2.8.0+cu128' 'numpy==2.5.2'
```

HF dependencies installed: transformers 5.23.0, peft 0.21.3, accelerate 1.15.0, safetensors 0.8.0; transitive packages include huggingface-hub 2.4.0 and tokenizers 0.23.3, plus their CLI/HTTP dependencies. `pip check` passed. Imports and actual CPU operations were tested before recording the versions in requirements-multimodal.txt; these are tested versions, not arbitrary proposed pins. Existing torch 2.8.0+cu128, CUDA runtime 12.8 and numpy 2.5.2 were reused. System Python still has no transformers installation.

bitsandbytes remains **uninstalled** and is isolated in requirements-multimodal-gpu.txt. Its optional install/compatibility is not claimed validated. Before any later GPU dependency install, inspect pip's plan using a constraint that preserves the actual torch build; do not run an unrestricted upgrade of GPU packages. No model weights or EEG datasets were downloaded; tests only save a tiny randomly initialized model/tokenizer into a pytest temporary directory.

## Environment and resource preflight

`collect_environment(cache_dir=None)` reports Python/library versions, CUDA/GPU facts and cache-filesystem free space. It never enumerates environment variables, reads credentials or prints HF tokens. Installed-version checks are labeled as such, not proof of native quantization support.

Preflight reports and expected CLI configuration/backbone error messages additionally redact recognizable HF/GitHub/AWS token syntax, URL userinfo and credential query/assignment values, including when embedded in local cache/model/config paths. Actual paths used for loading and filesystem inspection are not rewritten. This is defense against recognizable secret patterns, not a classifier for arbitrary secrets disguised as ordinary words; credentials must never be placed in paths or configuration values.

`run_preflight(config)` defaults to environment-only; no AutoConfig, tokenizer or model is accessed. `inspect_metadata=True` explicitly enables model-config inspection. `load_model=True` explicitly loads and probes a selected model. The CLI mirrors these with `--inspect-model` and `--load-model`. Actual network access additionally requires **both** local_files_only=false and `--allow-download` (or allow_download=True); an isolated download flag alone does not override local-only configuration. Default unit tests run with HF offline flags.

Actual command from `medley_pd/`, with no model-loading flags:

```bash
/tmp/medley-task1-tests/bin/python -m multimodal.llm_preflight --config configs/multimodal/fusion_b.json
```

Observed facts:

| Check | Actual result |
|---|---|
| Python / PyTorch | 3.12.3 / 2.8.0+cu128 |
| Transformers / PEFT / Accelerate | 5.23.0 / 0.21.3 / 1.15.0 |
| safetensors / bitsandbytes | 0.8.0 / missing |
| CUDA / GPU | available, runtime 12.8; NVIDIA GeForce RTX 4090 |
| CUDA capability / BF16 | 8.9 / supported |
| CUDA-reported VRAM | 25,252,724,736 total bytes; 24,838,406,144 free at observation (not a model measurement) |
| Model cache | `/workspace/.cache/huggingface/hub` |
| Cache filesystem | 53,687,091,200 total bytes; 49,914,896,384 free at final review observation |
| Selected production model | null; no pretrained loading attempted |
| Actual NF4 GPU forward/backward | not run |
| Actual model peak allocated/reserved | null, not measured |

`/workspace` persistence remains unconfirmed from prior environment inspection; free space is not evidence of durable storage. Preflight does not create cache directories, download weights, optimize models or start training.

### Estimates versus measurements

`estimate_parameter_storage(4_000_000_000)` gives hypothetical **weight-only** estimates: FP32 16 GB decimal, BF16/FP16 8 GB and packed NF4 lower bound 2 GB. The 4B count is a caller/default hypothetical, not inferred from an unselected model. Quantization metadata/unquantized layers, LoRA, optimizer state, activations, attention, CUDA workspaces and allocator overhead are excluded. `fit_in_24gb_verified` is False. No claim that a 4B model fits or trains follows from these numbers.

With explicit real-model load enabled, preflight can run three synthetic token positions through IDs and external embeddings, measure input gradients/base freezing and record actual CUDA peak allocations. That path was not executed on pretrained or quantized GPU weights in this task. The forward interface currently requests the causal LM's hidden-state output (and can incur its normal logits/intermediate-state overhead); it is not a memory-optimized 4B benchmark.

## CPU tests and smoke results

Executed from repository root:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q tests medley_pd/tests/test_configs.py medley_pd/tests/test_multimodal_config.py medley_pd/tests/test_multimodal_data_adapter.py medley_pd/tests/test_multimodal_dataset.py medley_pd/tests/test_multimodal_llm_backbone.py medley_pd/tests/test_multimodal_llm_preflight.py
```

Final full suite: **278 passed, 0 failed, 1 skipped (10.00s)**, including all 233 prior cases plus 45 passing Task 4 cases. The one skip is explicitly opt-in actual QLoRA GPU execution, requiring CUDA, bitsandbytes and MEDLEY_TEST_CACHED_LLM pointing to a local model directory; it is not a passing quantization test. Python compileall, the repository scaffold check and pip check passed. No dependency installation or GPU-stack change was performed during final review.

Final review reproduced four failures with five new targeted cases: Inf LoRA gradients were counted as a valid nonzero gradient; nonfinite final hidden states passed shape validation; and cache/model paths could expose recognizable credential patterns. The missing-bitsandbytes import case already passed. The initial command `CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/medley-task1-tests/bin/python -m pytest -q medley_pd/tests/test_multimodal_llm_backbone.py medley_pd/tests/test_multimodal_llm_preflight.py -k review` returned 4 failed, 1 passed, 40 deselected before corrections. All new cases pass in the full suite after correction.

A sixth review case reproduced the same credential-path disclosure in an expected CLI config-read error (1 failed, 5 deselected with `-k cli_error_paths` before correction). Error strings now use the report sanitizer as well. The final 278-pass suite includes this regression; no tests were removed or weakened.

CPU smoke coverage uses real locally instantiated tiny GPT2, Qwen2 and Gemma models and a tiny local WordLevel tokenizer. Frozen and LoRA modes both return `[1,3,16]` hidden states from IDs and inputs_embeds. External embedding gradients are finite/nonzero; base parameter gradients are absent. LoRA parameters receive nonzero gradients while base weights stay frozen. A training-mode non-reentrant checkpoint test preserves input gradients through Frozen mode. No optimizer, MoCA loss or downstream model is involved.

The probe now requires every produced LoRA gradient to be finite before `lora_nonzero_gradient_observed` can be True; it also exposes `lora_gradients_finite` (null in Frozen mode). Thus an Inf gradient is not evidence of valid trainability. Nonzero finite gradients are an observed synthetic backward result, not an inference from parameter counts.

Other tests cover batch size >1, both padding sides, real-token position equivalence, explicit position validation, internal mask holes, input exclusivity, malformed IDs/embeddings/masks, context overflow, tokenizer padding/EOS policy, target-module discovery, multimodal/encoder-decoder rejection, local synthetic save/load, effective snapshot revision/padding settings, strict optional config validation, old snapshot default injection, and no-default-network/no-secret preflight. NF4 config assertions use explicitly mocked backend/version prerequisites and prove only config construction. Full-precision model injection into QLoRA fails. Tiny Gemma tests do not prove MedGemma compatibility; CPU LoRA tests do not prove GPU NF4 backward compatibility.

PEFT emits a documented Conv1D fan_in_fan_out adjustment for GPT2 and some Transformers checkpointing deprecation warnings. They were not suppressed, and tests passed. This is not a claim of compatibility with untested future library versions.

## Usage and subsequent prerequisites

```python
from multimodal.config import load_multimodal_config, save_resolved_config
from multimodal.llm_backbone import load_llm_backbone, select_prediction_state

# Use a verified, already local text-only causal model directory/repository.
config = load_multimodal_config("configs/multimodal/fusion_b.json", [
    "llm.model_id=/private/verified-local-model",
    "llm.tuning=frozen", "llm.quantization=none",
])
backbone = load_llm_backbone(config)  # local-only; no automatic download
# A later serializer supplies synthetic/authorized text. Attention mask required.
# hidden = backbone.forward_hidden(input_ids=ids, attention_mask=mask)
# context = select_prediction_state(hidden, mask, positions=prediction_positions)
save_resolved_config(backbone.resolved_config, "/private/existing-run/resolved.json")
```

The paths above are illustrative, not existing/verified resources. Equivalent API loading was tested only using locally saved tiny synthetic models. For standard LoRA set tuning=lora and quantization=none; QLoRA retains the initial qlora/nf4 defaults but requires validated optional GPU dependencies and a compatible selected model.

Before Task 5: retain the allowlisted clinical/missing-value contract and select a tokenizer/template without leaking labels. Before Task 6/8: choose exact model ID/revision and access/license policy; validate actual architecture, tokenizer, sequence budget, inputs_embeds, PEFT targets and gradient flow; perform explicitly enabled, resource-bounded pretrained/NF4 tests and measure memory. Later trainable projector modules must preserve autograd through Frozen mode, and later regression heads must select a meaningful prediction position instead of assuming the final padded column. No such serializer, prediction suffix, projector, fusion or head is implemented here.

Real patient-data integration remains **NOT VERIFIED**. Earlier adapters/Dataset/Collator are covered by synthetic regressions and source-contract review only. MedGemma, actual pretrained 4B loading and actual quantized GPU forward/backward remain **NOT VERIFIED**; bitsandbytes is still absent in the observed environment. These are documented pre-training prerequisites, not claims established by the successful CPU tests or GPU environment check.

References checked for implementation guidance: [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization), [PEFT LoRA](https://huggingface.co/docs/peft/developer_guides/lora), [Transformers bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes). Runtime claims above derive from executed local tests and the observed environment, not from documentation alone.
