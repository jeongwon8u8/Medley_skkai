"""Architecture B genuine tiny causal forwards/backwards; synthetic CPU only."""
from dataclasses import replace
import copy
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
from multimodal.config import CONFIG_DIR, load_multimodal_config, save_resolved_config
from multimodal.clinical_tokenization import ClinicalTokenizer
from multimodal.llm_backbone import BackboneError, prepare_llm_backbone, select_prediction_state
from multimodal.fusion_b import EEGClinicalSoftTokenRegressor, assemble_soft_token_inputs


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def backbone(mode="frozen", side="right", family="gpt2", overrides=()):
    cfg = load_multimodal_config(CONFIG_DIR / "fusion_b.json", [f"llm.tuning={mode}",
        f"llm.quantization={'nf4' if mode == 'qlora' else 'none'}", "llm.compute_dtype=float32", "llm.device=cpu",
        f"llm.padding_side={side}", "llm.lora_dropout=0", "training.gradient_checkpointing=false", *overrides])
    words = ["[PAD]", "[UNK]", "[EOS]", "Predicted", "MoCA", ":", "The", "recorded", "age", "value", "is",
             ".", "patient", "'", "s", "sex", "female", "male", "UPDRS", "Part", "III", "score",
             "disease", "duration", "LEDD", "PIGD", "TD", *map(str, range(101))]
    backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    torch.manual_seed(29)
    if family == "gpt2":
        lm = GPT2LMHeadModel(GPT2Config(vocab_size=len(words), n_embd=16, n_head=2, n_layer=1, n_positions=256,
            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, bos_token_id=2, eos_token_id=2, pad_token_id=0))
    else:
        config_class, model_class = (Qwen2Config, Qwen2ForCausalLM) if family == "qwen2" else (GemmaConfig, GemmaForCausalLM)
        lm = model_class(config_class(vocab_size=len(words), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
            bos_token_id=2, eos_token_id=2, pad_token_id=0, attention_dropout=0))
    lm.set_attn_implementation("eager")
    if cfg.settings["llm"]["compute_dtype"] != "float32":
        lm.to(dtype=getattr(torch, cfg.settings["llm"]["compute_dtype"]))
    return prepare_llm_backbone(cfg, lm, tok)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("k", [1, 4, 7])
def test_a21_sequence_order_mask_suffix_indices_and_no_mutation(side, k):
    llm = backbone(side=side)
    text = ClinicalTokenizer(llm)(["The recorded age value is 68.", ""])
    prefix = torch.randn(2, k, 16, requires_grad=True)
    original_ids, original_mask, original_prefix = text.input_ids.clone(), text.attention_mask.clone(), prefix.detach().clone()
    fused = assemble_soft_token_inputs(llm, prefix, text)
    assert fused.inputs_embeds.shape == (2, k + text.input_ids.shape[1], 16)
    torch.testing.assert_close(fused.inputs_embeds[:, :k], prefix)
    torch.testing.assert_close(fused.inputs_embeds[:, k:], llm.embedding_layer(text.input_ids))
    assert torch.equal(fused.attention_mask[:, :k], torch.ones(2, k, dtype=torch.long))
    assert torch.equal(fused.attention_mask[:, k:], text.attention_mask)
    assert torch.equal(fused.prediction_positions, k + text.prediction_positions)
    suffix_ids = llm.tokenizer.encode("Predicted MoCA:", add_special_tokens=False)
    assert len(suffix_ids) == 3
    for row, pos in enumerate(fused.prediction_positions.tolist()):
        assert fused.attention_mask[row, pos] == 1
        assert text.input_ids[row, pos - k] == suffix_ids[-1]
        actual = text.input_ids[row, text.attention_mask[row].bool()].tolist()
        assert actual[-3:] == suffix_ids and actual.count(suffix_ids[0]) == 1
    fused.inputs_embeds.sum().backward()
    assert torch.equal(prefix.grad, torch.ones_like(prefix))
    assert torch.equal(text.input_ids, original_ids) and torch.equal(text.attention_mask, original_mask)
    assert torch.equal(prefix.detach(), original_prefix)


@pytest.mark.parametrize("case", ["width", "batch", "empty_k", "integer", "nonfinite", "device", "ids",
    "mask", "all_masked", "positions", "missing_positions", "padding_position", "budget"])
def test_a21_invalid_sequence_inputs(case):
    llm = backbone()
    text = ClinicalTokenizer(llm)(["age 68", ""])
    prefix = torch.randn(2, 4, 16)
    if case == "width": prefix = prefix[:, :, :15]
    elif case == "batch": prefix = prefix[:1]
    elif case == "empty_k": prefix = prefix[:, :0]
    elif case == "integer": prefix = prefix.long()
    elif case == "nonfinite": prefix[0, 0, 0] = float("nan")
    elif case == "device": prefix = prefix.to("meta")
    elif case == "ids": text.input_ids[0, 0] = -1
    elif case == "mask": text.attention_mask[0, 0] = 2
    elif case == "all_masked": text.attention_mask[0] = 0
    elif case == "positions": text = replace(text, prediction_positions=torch.zeros(2, dtype=torch.long))
    elif case == "missing_positions": text = replace(text, prediction_positions=None)
    elif case == "padding_position": text = replace(text, prediction_positions=torch.tensor([text.input_ids.shape[1]-1]*2))
    elif case == "budget": prefix = torch.randn(2, 255, 16)
    with pytest.raises((ValueError, BackboneError)):
        assemble_soft_token_inputs(llm, prefix, text)


def inputs(batch=3, width=3800):
    clinical = [dict(age=68., sex=None, updrs_part_iii=24., disease_duration=6.5, ledd=None),
                dict(age=72., sex=None, updrs_part_iii=None, disease_duration=None, ledd=None),
                dict.fromkeys(["age", "sex", "updrs_part_iii", "disease_duration", "ledd"])][:batch]
    torch.manual_seed(88)
    eeg = torch.randn(batch, 3, width)
    mask = torch.arange(3).unsqueeze(0) < torch.arange(1, batch+1).unsqueeze(1)
    return dict(eeg_features=eeg, eeg_mask=mask, clinical=clinical, artifact_flags=mask.clone())


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("batch", [1, 3])
def test_a22_forward_single_shared_inputs_embeds_pass_and_m1_representation(side, batch):
    llm = backbone(side=side)
    model = EEGClinicalSoftTokenRegressor(llm).eval()
    data = inputs(batch)
    before = copy.deepcopy(data)
    records, hidden_states, head_inputs = [], [], []
    handle = llm.model.register_forward_pre_hook(lambda module, args, kwargs: records.append(kwargs), with_kwargs=True)
    output_handle = llm.model.register_forward_hook(lambda module, args, output: hidden_states.append(output.hidden_states[-1]))
    head_handle = model.head.register_forward_pre_hook(lambda module, args: head_inputs.append(args[0]))
    predictions = model(**data)
    handle.remove()
    output_handle.remove()
    head_handle.remove()
    assert predictions.shape == (batch,) and torch.isfinite(predictions).all()
    assert len(records) == 1 and records[0]["input_ids"] is None
    fused = model.prepare_inputs(**data)
    torch.testing.assert_close(records[0]["inputs_embeds"], fused.inputs_embeds)
    assert torch.equal(records[0]["attention_mask"], fused.attention_mask)
    valid_positions = fused.attention_mask.cumsum(dim=1).sub(1).clamp_min(0)
    assert torch.equal(records[0]["position_ids"], valid_positions)
    # Existing M1 serializer/tokenizer reused verbatim, not a new B prompt policy.
    expected_text = model.serializer.serialize_batch(data["clinical"])
    text = ClinicalTokenizer(llm)(expected_text)
    k = model.pooling.num_queries
    torch.testing.assert_close(fused.inputs_embeds[:, k:], llm.embedding_layer(text.input_ids))
    assert torch.equal(fused.prediction_positions, text.prediction_positions + k)
    actual_selected = hidden_states[0][torch.arange(batch), k + text.prediction_positions]
    torch.testing.assert_close(head_inputs[0], actual_selected.float())
    assert "MoCA" not in expected_text[0]  # Only suffix names the target, never its value.
    for key in data:
        if isinstance(data[key], torch.Tensor):
            assert torch.equal(data[key], before[key])
        else:
            assert data[key] == before[key]
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(42.)
    assert torch.equal(model(**data), torch.full((batch,), 42.))


@pytest.mark.parametrize("encoder,width,k", [("cbramod", 3800, 1), ("scerebro", 3420, 7)])
def test_a22_width_k_and_a1_injection(encoder, width, k):
    from multimodal.eeg_pooling import EEGFeatureAdapter, LearnedQueryAttentionPooling
    from multimodal.eeg_projection import EEGToLLMProjector
    llm = backbone(overrides=[f"eeg.encoder={encoder}", f"eeg.feature_dim={width}", f"eeg.pooling.num_queries={k}",
                              "eeg.pooling.hidden_dim=32", "eeg.pooling.num_heads=8"])
    cfg = llm.resolved_config
    adapter, pool = EEGFeatureAdapter.from_config(cfg), LearnedQueryAttentionPooling.from_config(cfg)
    projector = EEGToLLMProjector.from_config(cfg, hidden_dim=llm.hidden_dim)
    model = EEGClinicalSoftTokenRegressor(llm, pooling=pool, projector=projector, feature_adapter=adapter)
    assert model.pooling is pool and model.projector is projector and model.feature_adapter is adapter
    data = inputs(width=width)
    text = model.tokenizer(model.serializer.serialize_batch(data["clinical"]))
    fused = model.prepare_inputs(**data)
    assert fused.inputs_embeds.shape == (3, k + text.input_ids.shape[1], 16)
    assert torch.equal(fused.attention_mask[:, :k], torch.ones(3, k, dtype=torch.long))
    assert model(**data).shape == (3,)


@pytest.mark.parametrize("key", ["moca", "MoCA_score", "ctt", "updrs_total", "subject_id", "subgroup", "visit_id"])
def test_a22_contaminated_clinical_rejected_before_llm(key):
    llm = backbone()
    model = EEGClinicalSoftTokenRegressor(llm)
    data = inputs()
    data["clinical"][1][key] = "SECRET_VALUE"
    calls = []
    handle = llm.model.register_forward_pre_hook(lambda *args: calls.append(1))
    with pytest.raises(ValueError) as exc:
        model(**data)
    handle.remove()
    assert not calls and "SECRET_VALUE" not in str(exc.value)


@pytest.mark.parametrize("case", ["missing_mask", "wrong_width", "all_masked", "clinical_batch", "nonbool_mask"])
def test_a22_invalid_model_inputs(case):
    model = EEGClinicalSoftTokenRegressor(backbone())
    data = inputs()
    if case == "missing_mask": del data["eeg_mask"]
    elif case == "wrong_width": data["eeg_features"] = data["eeg_features"][:, :, :10]
    elif case == "all_masked": data["eeg_mask"][0] = False
    elif case == "clinical_batch": data["clinical"] = data["clinical"][:1]
    elif case == "nonbool_mask": data["eeg_mask"] = data["eeg_mask"].float()
    with pytest.raises((ValueError, TypeError)):
        model(**data)


@pytest.mark.parametrize("family", ["gpt2", "qwen2", "gemma"])
@pytest.mark.parametrize("mode", ["frozen", "lora"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_a23_genuine_backward_through_llm_to_eeg_and_clinical_context(family, mode, side):
    from multimodal.clinical_regressor import moca_huber_loss
    model = EEGClinicalSoftTokenRegressor(backbone(mode, side, family)).train()
    data = inputs()
    data["eeg_features"].requires_grad_(True)
    fused_embeddings = []

    def retain_fused(module, args, kwargs):
        kwargs["inputs_embeds"].retain_grad()
        fused_embeddings.append(kwargs["inputs_embeds"])

    handle = model.llm.register_forward_pre_hook(retain_fused, with_kwargs=True)
    loss = moca_huber_loss(model(**data), torch.tensor([19.125, 20.125, 21.125]))
    loss.backward()
    handle.remove()
    assert torch.isfinite(loss)
    for prefix, module in (("pool", model.pooling), ("projector", model.projector), ("head", model.head)):
        for name, p in module.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"{prefix}.{name}"
            assert p.grad.abs().sum() > 0, f"{prefix}.{name}"
    assert data["eeg_features"].grad[data["eeg_mask"]].abs().sum() > 0
    assert torch.equal(data["eeg_features"].grad[~data["eeg_mask"]],
                       torch.zeros_like(data["eeg_features"].grad[~data["eeg_mask"]]))
    grad = fused_embeddings[0].grad
    k = model.pooling.num_queries
    assert grad[:, :k].abs().sum() > 0  # Actual causal path to prefix EEG embeddings.
    text = model.tokenizer(model.serializer.serialize_batch(data["clinical"]))
    assert grad[0, k:k+5].abs().sum() > 0  # Preceding clinical tokens influence suffix output.
    assert torch.equal(grad[:, k:][~text.attention_mask.bool()],
                       torch.zeros_like(grad[:, k:][~text.attention_mask.bool()]))
    base = [p for n, p in model.llm.named_parameters() if "lora_" not in n]
    assert all(not p.requires_grad and p.grad is None for p in base)
    adapters = [p for n, p in model.llm.named_parameters() if "lora_" in n]
    if mode == "lora":
        assert adapters and all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in adapters)
        assert any(p.grad.abs().sum() > 0 for p in adapters)
    else:
        assert not adapters


@pytest.mark.parametrize("family", ["gpt2", "qwen2", "gemma"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_a23_padding_batch_equivalence_and_real_eeg_influence(family, side):
    model = EEGClinicalSoftTokenRegressor(backbone(side=side, family=family)).eval()
    data = inputs()
    batched = model(**data)
    for row in range(3):
        single = {k: v[row:row+1] for k, v in data.items()}
        torch.testing.assert_close(model(**single)[0], batched[row], atol=2e-5, rtol=2e-5)
    changed = {**data, "eeg_features": data["eeg_features"].clone()}
    changed["eeg_features"][data["eeg_mask"]] += 3.
    assert not torch.allclose(model(**changed), batched, rtol=0, atol=1e-8)
    padded = {**data, "eeg_features": data["eeg_features"].clone(), "artifact_flags": ~data["artifact_flags"]}
    padded["eeg_features"][~data["eeg_mask"]] = float("nan")
    torch.testing.assert_close(model(**padded), batched, rtol=0, atol=0)


def test_a23_frozen_checkpointing_preserves_eeg_gradients():
    model = EEGClinicalSoftTokenRegressor(backbone(overrides=["training.gradient_checkpointing=true"])).train()
    model(**inputs()).sum().backward()
    assert torch.isfinite(model.pooling.queries.grad).all() and model.pooling.queries.grad.abs().sum() > 0
    assert model.projector.linear.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_a23_fp32_eeg_to_actual_embedding_dtype_without_llm_cast(dtype):
    llm = backbone(overrides=[f"llm.compute_dtype={dtype}"])
    model = EEGClinicalSoftTokenRegressor(llm)
    expected = getattr(torch, dtype)
    assert model.pooling.input_projection.weight.dtype == torch.float32
    assert model.projector.linear.weight.dtype == torch.float32
    assert llm.embedding_layer.weight.dtype == expected
    fused = model.prepare_inputs(**inputs())
    assert fused.inputs_embeds.dtype == expected
    model(**inputs()).sum().backward()
    assert model.projector.linear.weight.grad is not None and torch.isfinite(model.projector.linear.weight.grad).all()
    assert model.pooling.queries.grad.abs().sum() > 0
    assert llm.embedding_layer.weight.dtype == expected


def test_a23_qlora_cpu_has_no_fallback():
    with pytest.raises(BackboneError):
        backbone("qlora")


@pytest.mark.parametrize("feature_set", ["C0", "C1", "C2"])
def test_a24_clinical_set_and_effective_config_snapshot_compatibility(feature_set, tmp_path):
    model = EEGClinicalSoftTokenRegressor(backbone(overrides=[f"clinical_set={feature_set}"]))
    data = inputs()
    fields = model.backbone.resolved_config.clinical_variables
    data["clinical"] = [{key: record.get(key) for key in fields} for record in data["clinical"]]
    assert model(**data).shape == (3,)
    snapshot = tmp_path / "resolved.json"
    save_resolved_config(model.backbone.resolved_config, snapshot)
    assert load_multimodal_config(snapshot).to_dict() == model.backbone.resolved_config.to_dict()


def test_a24_actual_task3_batch_smoke_label_metadata_separation():
    import numpy as np
    from types import MappingProxyType, SimpleNamespace
    from multimodal.data_adapter import SubjectRecord
    from multimodal.dataset import MedleySubjectDataset, MedleySubjectCollator

    model = EEGClinicalSoftTokenRegressor(backbone()).eval()
    data = inputs()
    ids = ("synthetic-a", "synthetic-b", "synthetic-c")
    records = {sid: SubjectRecord(sid, data["eeg_features"][i, :i+1].numpy(), np.ones(i+1, dtype=bool),
        MappingProxyType(data["clinical"][i]), 19.125+i, "SYNTHETIC_SUBGROUP") for i, sid in enumerate(ids)}
    adapter = SimpleNamespace(subject_ids=ids, clinical_variables=model.backbone.resolved_config.clinical_variables,
                              get_subject=records.__getitem__)
    dataset = MedleySubjectDataset(adapter, ids)
    batch = MedleySubjectCollator()([dataset[i] for i in range(3)])
    before = copy.deepcopy(batch)

    def predict():
        return model(eeg_features=batch["eeg_features"], eeg_mask=batch["eeg_mask"],
                     clinical=batch["clinical"], artifact_flags=batch["artifact_flags"])

    predictions = predict()
    assert batch["subject_ids"] == list(ids) and predictions.shape == (3,)
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            assert torch.equal(batch[key], before[key])
        else:
            assert batch[key] == before[key]
    texts = model.serializer.serialize_batch(batch["clinical"])
    assert all(not any(secret in text for secret in (*ids, "SYNTHETIC_SUBGROUP", "19.125", "20.125", "21.125"))
               for text in texts)
    batch["moca"].fill_(29.875)
    batch["subject_ids"].reverse()
    batch["subgroups"] = ["OTHER_METADATA"] * 3
    torch.testing.assert_close(predict(), predictions, atol=0, rtol=0)
    with pytest.raises(TypeError):
        model(**data, moca=batch["moca"])


@pytest.mark.parametrize("case", ["architecture", "adapter_on", "projection_config", "pool_k", "projector_width", "base_trainable"])
def test_a24_invalid_or_unsupported_model_configuration(case):
    from multimodal.eeg_pooling import LearnedQueryAttentionPooling
    from multimodal.eeg_projection import EEGToLLMProjector
    llm = backbone()
    kwargs = {}
    if case == "architecture":
        llm.resolved_config.settings["architecture"] = "clinical_only"
        llm.resolved_config.settings["fusion"]["type"] = "none"
    elif case == "adapter_on": llm.resolved_config.settings["eeg"]["feature_adapter"]["enabled"] = True
    elif case == "projection_config": llm.resolved_config.settings["eeg"]["projector"]["output_dim"] = 32
    elif case == "pool_k": kwargs["pooling"] = LearnedQueryAttentionPooling(3800, num_queries=7)
    elif case == "projector_width": kwargs["projector"] = EEGToLLMProjector(256, 32)
    elif case == "base_trainable": llm.embedding_layer.weight.requires_grad_(True)
    with pytest.raises((ValueError, NotImplementedError)):
        EEGClinicalSoftTokenRegressor(llm, **kwargs)


def test_a24_model_context_budget_includes_eeg_prefix():
    model = EEGClinicalSoftTokenRegressor(backbone(overrides=["llm.max_text_tokens=3", "llm.max_sequence_tokens=5"]))
    data = inputs(batch=1)
    data["clinical"] = [dict.fromkeys(data["clinical"][0])]
    with pytest.raises(ValueError, match="EEG prefix"):
        model(**data)


def test_a24_seeded_fresh_models_never_share_fitted_m1_weights():
    from multimodal.clinical_regressor import ClinicalLLMRegressor
    from multimodal.eeg_pooling import seed_eeg_modules
    llm1, llm2 = backbone(), backbone()
    seed_eeg_modules(llm1.resolved_config)
    first = EEGClinicalSoftTokenRegressor(llm1)
    seed_eeg_modules(llm2.resolved_config)
    second = EEGClinicalSoftTokenRegressor(llm2)
    assert first.llm is not second.llm
    for a, b in zip(first.parameters(), second.parameters()):
        assert a.data_ptr() != b.data_ptr()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    llm3 = backbone(overrides=["architecture=clinical_only", "fusion.type=none"])
    m1 = ClinicalLLMRegressor(llm3)
    assert first.head.weight.data_ptr() != m1.head.weight.data_ptr()
    old = first.head.weight.detach().clone()
    with torch.no_grad():
        m1.head.weight.fill_(42)
    assert torch.equal(old, first.head.weight)


@pytest.mark.parametrize("side", ["left", "right"])
def test_a24_eos_as_pad_still_selects_actual_suffix(side):
    llm = backbone(side=side)
    llm.tokenizer.pad_token = llm.tokenizer.eos_token
    llm.model.config.pad_token_id = llm.tokenizer.eos_token_id
    model = EEGClinicalSoftTokenRegressor(llm)
    data = inputs()
    text = model.tokenizer(model.serializer.serialize_batch(data["clinical"]))
    fused = model.prepare_inputs(**data)
    k = model.pooling.num_queries
    for row, pos in enumerate(fused.prediction_positions.tolist()):
        assert fused.attention_mask[row, pos] == 1
        assert text.input_ids[row, pos-k] != llm.tokenizer.eos_token_id
    assert torch.isfinite(model(**data)).all()


def test_a24_new_eeg_modules_stay_fp32_under_changed_global_default_dtype():
    llm = backbone()
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = EEGClinicalSoftTokenRegressor(llm)
    finally:
        torch.set_default_dtype(previous)
    assert model.pooling.input_projection.weight.dtype == torch.float32
    assert model.projector.linear.weight.dtype == torch.float32
    assert model.head.weight.dtype == torch.float32
    assert llm.embedding_layer.weight.dtype == torch.float32
    assert torch.isfinite(model(**inputs())).all()


def test_a24_explicit_dtype_conversion_overflow_rejected_before_llm():
    llm = backbone(overrides=["llm.compute_dtype=float16"])
    text = ClinicalTokenizer(llm)([""])
    with pytest.raises(ValueError, match="after explicit dtype conversion"):
        assemble_soft_token_inputs(llm, torch.full((1, 4, 16), 1e10), text)
