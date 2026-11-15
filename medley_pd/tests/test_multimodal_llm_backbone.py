"""Tiny locally instantiated HF models; no downloaded weights or patient text."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (GPT2Config, GPT2LMHeadModel, Qwen2Config, Qwen2ForCausalLM,
                          GemmaConfig, GemmaForCausalLM, PreTrainedTokenizerFast)

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from multimodal.config import (CONFIG_DIR, ConfigError, LLM_RUNTIME_DEFAULTS, load_multimodal_config,
                               save_resolved_config)  # noqa: E402
from multimodal.llm_backbone import (BackboneError, inspect_model_config, load_llm_backbone,
    parameter_statistics, prepare_llm_backbone, quantization_config, require_quantization_backend,
    resolve_lora_targets, select_prediction_state)  # noqa: E402
from multimodal.llm_preflight import probe_backbone  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def configuration(mode="frozen", overrides=()):
    return load_multimodal_config(CONFIG_DIR / "fusion_b.json", [f"llm.tuning={mode}",
        f"llm.quantization={'nf4' if mode == 'qlora' else 'none'}", "llm.compute_dtype=float32", "llm.device=cpu", *overrides])


def tokenizer():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "hello": 3, "synthetic": 4, "text": 5}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]")


def tiny_model(family="gpt2"):
    torch.manual_seed(42)
    if family == "gpt2":
        cfg = GPT2Config(vocab_size=16, n_embd=16, n_layer=1, n_head=2, n_positions=32,
                         bos_token_id=2, eos_token_id=2, pad_token_id=0,
                         resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
        model = GPT2LMHeadModel(cfg)
    else:
        cls, model_cls = (Qwen2Config, Qwen2ForCausalLM) if family == "qwen2" else (GemmaConfig, GemmaForCausalLM)
        cfg = cls(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                  num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                  max_position_embeddings=32, bos_token_id=2, eos_token_id=2, pad_token_id=0)
        model = model_cls(cfg)
    model.set_attn_implementation("eager")
    return model


@pytest.mark.parametrize("family", ["gpt2", "qwen2", "gemma"])
@pytest.mark.parametrize("mode", ["frozen", "lora"])
def test_tiny_backbone_modes_shapes_and_backward(family, mode):
    backbone = prepare_llm_backbone(configuration(mode), tiny_model(family), tokenizer())
    assert backbone.hidden_dim == 16 and backbone.mode == mode
    assert backbone.metadata()["model_type"] == family
    assert backbone.embedding_layer is backbone.model.get_input_embeddings()
    report = probe_backbone(backbone)
    assert report["input_ids_hidden_shape"] == report["inputs_embeds_hidden_shape"] == [1, 3, 16]
    assert report["external_embedding_gradients_finite_nonzero"]
    assert report["base_gradients_absent"]
    stats = parameter_statistics(backbone.model)
    assert stats["base_frozen"]
    if mode == "frozen":
        assert stats["trainable_parameters"] == 0 and not hasattr(backbone.model, "peft_config")
    else:
        assert stats["trainable_parameters"] == stats["trainable_lora_parameters"] > 0
        assert report["lora_nonzero_gradient_observed"]


def test_training_checkpointing_does_not_block_external_gradients():
    backbone = prepare_llm_backbone(configuration(), tiny_model(), tokenizer())
    backbone.model.train()
    assert backbone.model.is_gradient_checkpointing
    assert probe_backbone(backbone)["external_embedding_gradients_finite_nonzero"]


@pytest.mark.parametrize("side", ["left", "right"])
def test_synthetic_tokenization_and_padding(side):
    backbone = prepare_llm_backbone(configuration(overrides=[f"llm.padding_side={side}"]), tiny_model(), tokenizer())
    tokens = backbone.tokenizer(["hello synthetic text", "hello"], padding=True, return_tensors="pt")
    hidden = backbone.forward_hidden(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
    assert hidden.shape == (2, 3, 16)
    selected = select_prediction_state(hidden, tokens["attention_mask"])
    single = backbone.forward_hidden(input_ids=torch.tensor([[3]]), attention_mask=torch.ones(1, 1, dtype=torch.long))
    torch.testing.assert_close(selected[1], single[0, 0], atol=1e-5, rtol=1e-5)


def test_prediction_positions_left_right_and_holes():
    hidden = torch.arange(2*5*3).reshape(2, 5, 3).float().requires_grad_(True)
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 0, 1, 0, 0]])
    result = select_prediction_state(hidden, mask)
    assert torch.equal(result, torch.stack([hidden[0, 4], hidden[1, 2]]))
    explicit = select_prediction_state(hidden, mask, torch.tensor([2, 0]))
    explicit.sum().backward()
    assert hidden.grad is not None
    for positions in (torch.tensor([1, 0]), torch.tensor([9, 0]), torch.tensor([2.0, 0.0])):
        with pytest.raises(BackboneError):
            select_prediction_state(hidden, mask, positions)
    with pytest.raises(BackboneError, match="unmasked"):
        select_prediction_state(hidden, torch.zeros(2, 5))


@pytest.mark.parametrize("case", ["both", "neither", "bad_ids", "bad_width", "nan", "empty_mask", "wrong_mask", "too_long"])
def test_forward_errors(case):
    backbone = prepare_llm_backbone(configuration(), tiny_model(), tokenizer())
    ids = torch.tensor([[3, 4]])
    kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if case == "both": kwargs["inputs_embeds"] = torch.ones(1, 2, 16)
    if case == "neither": kwargs.pop("input_ids")
    if case == "bad_ids": kwargs["input_ids"] = torch.tensor([[999, 1]])
    if case in ("bad_width", "nan"):
        kwargs.pop("input_ids")
        kwargs["inputs_embeds"] = torch.ones(1, 2, 15) if case == "bad_width" else torch.full((1, 2, 16), float("nan"))
    if case == "empty_mask": kwargs["attention_mask"] = torch.zeros_like(ids)
    if case == "wrong_mask": kwargs["attention_mask"] = torch.ones(1, 2)*2
    if case == "too_long": kwargs = {"input_ids": torch.ones(1, 40, dtype=torch.long), "attention_mask": torch.ones(1, 40)}
    with pytest.raises(BackboneError):
        backbone.forward_hidden(**kwargs)


def test_padding_policy_requires_explicit_eos_choice():
    tok = tokenizer()
    tok.pad_token = None
    with pytest.raises(BackboneError, match="pad token"):
        prepare_llm_backbone(configuration(), tiny_model(), tok)
    old_size = len(tok)
    backbone = prepare_llm_backbone(configuration(overrides=["llm.pad_token_policy=eos"]), tiny_model(), tok)
    assert len(tok) == old_size and tok.pad_token_id == tok.eos_token_id
    assert backbone.model.config.pad_token_id == tok.eos_token_id


def test_lora_targets_are_actual_modules_and_explicit_targets_checked():
    targets = resolve_lora_targets(tiny_model("qwen2"), ["q_proj", "v_proj"])
    assert targets and all(name.endswith(("q_proj", "v_proj")) for name in targets)
    with pytest.raises(BackboneError, match="target"):
        resolve_lora_targets(tiny_model(), ["does_not_exist"])


def test_multimodal_and_encoder_decoder_metadata_fail():
    cfg = GPT2Config()
    cfg.vision_config = {"test": True}
    with pytest.raises(BackboneError, match="text-only"):
        inspect_model_config(cfg)
    cfg.vision_config = None
    cfg.is_encoder_decoder = True
    with pytest.raises(BackboneError, match="decoder"):
        inspect_model_config(cfg)


def test_local_saved_synthetic_model_load_and_snapshot(tmp_path):
    root = tmp_path / "tiny-local"
    tiny_model().save_pretrained(root)
    tokenizer().save_pretrained(root)
    cfg = configuration(overrides=[f"llm.model_id={root}"])
    backbone = load_llm_backbone(cfg)
    assert backbone.metadata()["hidden_dim"] == 16
    assert probe_backbone(backbone)["external_embedding_gradients_finite_nonzero"]
    path = save_resolved_config(backbone.resolved_config, tmp_path / "resolved.json")
    restored = load_multimodal_config(path)
    assert restored.settings["llm"]["padding_side"] == "right"
    assert restored.settings["llm"]["device"] == "cpu"


def test_no_network_without_explicit_gate(monkeypatch):
    import transformers
    def prohibited(*args, **kwargs): raise AssertionError("Network/model access should be blocked before HF")
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", prohibited)
    cfg = configuration(overrides=["llm.model_id=synthetic/not-accessed", "llm.local_files_only=false"])
    with pytest.raises(BackboneError, match="allow_download"):
        load_llm_backbone(cfg)
    with pytest.raises(ConfigError, match="concrete"):
        load_llm_backbone(configuration())


def test_missing_quantization_backend_and_no_full_precision_qlora(monkeypatch):
    import multimodal.llm_backbone as module
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(BackboneError, match="CUDA"):
        require_quantization_backend()
    monkeypatch.setattr(module, "require_quantization_backend", lambda: None)
    with pytest.raises(BackboneError, match="genuinely"):
        prepare_llm_backbone(configuration("qlora"), tiny_model(), tokenizer())


def test_frozen_rejects_already_adapted_model():
    adapted = prepare_llm_backbone(configuration("lora"), tiny_model("qwen2"), tokenizer())
    with pytest.raises(BackboneError, match="unadapted"):
        prepare_llm_backbone(configuration(), adapted.model, tokenizer())


def test_snapshot_pins_resolved_revision_and_tokenizer_mismatch_fails():
    model = tiny_model()
    model.config._commit_hash = "a" * 40  # synthetic metadata, no remote revision claimed verified
    backbone = prepare_llm_backbone(configuration(), model, tokenizer())
    assert backbone.resolved_config.settings["llm"]["revision"] == "a" * 40
    assert backbone.metadata()["requested_revision"] is None
    tok = tokenizer()
    tok.add_tokens([f"synthetic-extra-{i}" for i in range(20)])
    with pytest.raises(BackboneError, match="vocabulary"):
        prepare_llm_backbone(configuration(), tiny_model(), tok)


def test_review_nonfinite_lora_gradients_are_not_reported_as_valid():
    backbone = prepare_llm_backbone(configuration("lora"), tiny_model("qwen2"), tokenizer())
    parameter = next(p for name, p in backbone.model.named_parameters() if "lora_B" in name)
    parameter.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    report = probe_backbone(backbone)
    assert report["lora_nonzero_gradient_observed"] is False


def test_review_nonfinite_hidden_states_fail(monkeypatch):
    from types import SimpleNamespace
    backbone = prepare_llm_backbone(configuration(), tiny_model(), tokenizer())
    monkeypatch.setattr(backbone.model, "forward", lambda **kwargs:
                        SimpleNamespace(hidden_states=(torch.full((1, 2, 16), float("nan")),)))
    with pytest.raises(BackboneError, match="finite"):
        backbone.forward_hidden(input_ids=torch.tensor([[3, 4]]), attention_mask=torch.ones(1, 2))


def test_review_missing_bitsandbytes_import_is_actionable(monkeypatch):
    import multimodal.llm_backbone as module
    original = module.importlib.import_module
    def missing(name):
        if name == "bitsandbytes":
            raise ModuleNotFoundError("synthetic missing backend")
        return original(name)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.importlib, "import_module", missing)
    with pytest.raises(BackboneError, match="bitsandbytes"):
        require_quantization_backend()


def test_nf4_configuration_only_with_mocked_backend(monkeypatch):
    # Pure config construction; neither CUDA kernels nor QLoRA execution proven.
    import multimodal.llm_backbone as module
    original = importlib.metadata.version
    monkeypatch.setattr(module, "require_quantization_backend", lambda: None)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.49.0" if name == "bitsandbytes" else original(name))
    cfg = quantization_config(configuration("qlora").settings["llm"])
    assert cfg.load_in_4bit and cfg.bnb_4bit_quant_type == "nf4" and cfg.bnb_4bit_use_double_quant
    assert cfg.bnb_4bit_compute_dtype == torch.float32


@pytest.mark.parametrize("override", ["llm.local_files_only=1", "llm.trust_remote_code=true",
    "llm.target_modules=[]", "llm.target_modules=[1]", "llm.max_sequence_tokens=0",
    "llm.padding_side=middle", "llm.pad_token_policy=add", "llm.device=invalid"])
def test_optional_config_validation(override):
    with pytest.raises(ConfigError): configuration(overrides=[override])


def test_old_snapshot_optional_defaults_and_unknown_keys(tmp_path):
    cfg = configuration()
    document = cfg.to_dict()
    for key in LLM_RUNTIME_DEFAULTS: document["settings"]["llm"].pop(key)
    path = tmp_path / "old-snapshot.json"
    path.write_text(json.dumps(document))
    restored = load_multimodal_config(path)
    assert restored.settings["llm"]["local_files_only"] is True
    with pytest.raises(ConfigError, match="Unknown"):
        configuration(overrides=["llm.unknown=true"])


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("bitsandbytes") is None
                    or not os.environ.get("MEDLEY_TEST_CACHED_LLM"),
                    reason="Actual GPU NF4 requires CUDA, bitsandbytes and explicit MEDLEY_TEST_CACHED_LLM local-model opt-in")
def test_opt_in_actual_qlora_gpu():
    path = Path(os.environ["MEDLEY_TEST_CACHED_LLM"])
    assert path.is_dir(), "GPU opt-in requires a local model directory; downloads remain disabled"
    cfg = configuration("qlora", [f"llm.model_id={path}", "llm.device=cuda:0", "llm.compute_dtype=bfloat16"])
    backbone = load_llm_backbone(cfg)
    report = probe_backbone(backbone)
    assert report["external_embedding_gradients_finite_nonzero"] and report["lora_nonzero_gradient_observed"]
    assert report["base_gradients_absent"]
