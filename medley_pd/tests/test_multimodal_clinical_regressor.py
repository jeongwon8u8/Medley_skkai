"""M1 tiny local causal models; CPU only, no pretrained downloads."""
from pathlib import Path
import sys

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from multimodal.config import CONFIG_DIR, ConfigError, load_multimodal_config
from multimodal.llm_backbone import prepare_llm_backbone, select_prediction_state, BackboneError
from multimodal.clinical_serializer import ClinicalTextSerializer
from multimodal.clinical_tokenization import ClinicalTokenizer
from multimodal.clinical_regressor import ClinicalLLMRegressor, moca_huber_loss


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(mode="frozen", side="right", overrides=()):
    cfg = load_multimodal_config(CONFIG_DIR / "clinical_only.json", [
        f"llm.tuning={mode}", f"llm.quantization={'nf4' if mode == 'qlora' else 'none'}",
        "llm.compute_dtype=float32", "llm.device=cpu", f"llm.padding_side={side}",
        "training.gradient_checkpointing=false", "llm.lora_dropout=0", *overrides])
    vocabulary = ["[PAD]", "[UNK]", "[EOS]", "Predicted", "MoCA", ":", ".", "The", "recorded", "age",
                  "value", "is", "patient", "'", "s", "sex", "female", "male", "UPDRS", "Part", "III",
                  "score", "disease", "duration", "LEDD", "PIGD", "TD", *map(str, range(101))]
    backend = Tokenizer(WordLevel(dict(zip(vocabulary, range(len(vocabulary)))), unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    torch.manual_seed(71)
    lm = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_embd=16, n_head=2, n_layer=1,
        n_positions=256, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, bos_token_id=2, eos_token_id=2, pad_token_id=0))
    lm.set_attn_implementation("eager")
    return ClinicalLLMRegressor(prepare_llm_backbone(cfg, lm, tokenizer))


def forward(model, tokens):
    return model(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask,
                 prediction_positions=tokens.prediction_positions)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("texts", [[""], ["The recorded age value is 68."],
                                   ["The recorded age value is 68.", "", "The recorded age value is 72."]])
def test_prediction_positions_padding_shapes_and_batch_independence(side, texts):
    model = make_model(side=side)
    model.eval()
    tokenize = ClinicalTokenizer(model.backbone)
    tokens = tokenize(texts)
    assert torch.equal(tokens.input_ids, tokenize(texts).input_ids)
    suffix_ids = model.backbone.tokenizer.encode("Predicted MoCA:", add_special_tokens=False)
    assert len(suffix_ids) == 3
    for row, position in enumerate(tokens.prediction_positions.tolist()):
        assert tokens.attention_mask[row, position] == 1
        assert tokens.input_ids[row, position] == suffix_ids[-1]
        actual = tokens.input_ids[row][tokens.attention_mask[row].bool()].tolist()
        assert actual[-3:] == suffix_ids and actual[-1] != model.backbone.tokenizer.eos_token_id
    predictions = forward(model, tokens)
    assert predictions.shape == (len(texts),) and torch.isfinite(predictions).all()
    assert model.head.in_features == 16 and model.head.out_features == 1
    separate = torch.cat([forward(model, tokenize([text])) for text in texts])
    torch.testing.assert_close(predictions, separate, rtol=1e-5, atol=1e-6)


def test_serializer_to_tokens_without_targets_and_unclamped_head():
    model = make_model()
    serializer = ClinicalTextSerializer(model.backbone.resolved_config)
    text = serializer.serialize(dict(age=68, sex="female", updrs_part_iii=24, disease_duration=6.5, ledd=None))
    assert "MoCA" not in text and "29.875" not in text
    tokens = ClinicalTokenizer(model.backbone)([text])
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(42)
    assert forward(model, tokens).item() == 42  # No sigmoid/clipping.
    with pytest.raises(TypeError):
        model(eeg_features=torch.zeros(1, 4))


def test_causal_suffix_representation_depends_on_clinical_prefix():
    model = make_model()
    model.eval()
    tokens = ClinicalTokenizer(model.backbone)(["The recorded age value is 68."])
    embeddings = model.backbone.embedding_layer(tokens.input_ids).detach().requires_grad_(True)
    hidden = model.backbone.forward_hidden(inputs_embeds=embeddings, attention_mask=tokens.attention_mask)
    state = select_prediction_state(hidden, tokens.attention_mask, tokens.prediction_positions)
    state[:, 0].sum().backward()
    assert embeddings.grad[:, :5].abs().sum() > 0
    other = ClinicalTokenizer(model.backbone)(["The recorded age value is 72."])
    assert not torch.allclose(forward(model, tokens), forward(model, other), atol=1e-8, rtol=0)


def test_position_validation():
    model = make_model()
    tokens = ClinicalTokenizer(model.backbone)(["age", "age age"])
    with pytest.raises((ValueError, BackboneError)):
        model(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask,
              prediction_positions=torch.zeros(2, dtype=torch.long))


@pytest.mark.parametrize("text_budget,total_budget,text", [(1, 1, ""), (2, 20, "age age age"), (3, 3, "age")])
def test_context_budget_never_truncates(text_budget, total_budget, text):
    model = make_model(overrides=[f"llm.max_text_tokens={text_budget}", f"llm.max_sequence_tokens={total_budget}"])
    with pytest.raises(ValueError):
        ClinicalTokenizer(model.backbone)([text])


@pytest.mark.parametrize("mode", ["frozen", "lora"])
def test_head_and_adapter_gradients_only(mode):
    model = make_model(mode)
    tokens = ClinicalTokenizer(model.backbone)(["age 68", "age 72"])
    loss = moca_huber_loss(forward(model, tokens), torch.tensor([21., 25.]))
    loss.backward()
    assert model.head.weight.grad is not None and model.head.weight.grad.abs().sum() > 0
    base = [p for n, p in model.llm.named_parameters() if "lora_" not in n]
    assert all(not p.requires_grad and p.grad is None for p in base)
    adapter_grads = [p.grad for n, p in model.llm.named_parameters() if "lora_" in n]
    if mode == "lora":
        assert adapter_grads and all(g is not None and torch.isfinite(g).all() for g in adapter_grads)
        assert any(g.abs().sum() > 0 for g in adapter_grads)


def test_huber_exact_and_target_range():
    pred = torch.tensor([0., 1., 5.], requires_grad=True)
    target = torch.tensor([0., 1.5, 3.])
    loss = moca_huber_loss(pred, target)
    torch.testing.assert_close(loss, torch.tensor((0 + .125 + 1.5) / 3))
    loss.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("pred,target", [(torch.ones(2), torch.ones(2, 1)),
    (torch.ones(2, 1), torch.ones(2, 1)), (torch.ones(1), torch.tensor([float('nan')])),
    (torch.tensor([float('inf')]), torch.ones(1)), (torch.ones(1), torch.tensor([31.])),
    (torch.ones(1), torch.tensor([-1.])), (torch.ones(1), torch.ones(1, dtype=torch.long)),
    (torch.empty(0), torch.empty(0))])
def test_invalid_loss_inputs(pred, target):
    with pytest.raises(ValueError):
        moca_huber_loss(pred, target)


@pytest.mark.parametrize("override", ["clinical_regression.head_type=mlp", "clinical_regression.random_seed=-1",
    "clinical_regression.mixed_precision=float16", "clinical_regression.prediction_suffix=answer"])
def test_invalid_m1_options(override):
    with pytest.raises(ConfigError):
        make_model(overrides=[override])


def test_no_mocked_qlora_or_fusion_mode():
    with pytest.raises(BackboneError):
        make_model("qlora")  # CPU path must fail, never silently use float LoRA.
    model = make_model()
    model.backbone.resolved_config.settings["architecture"] = "fusion_a"
    with pytest.raises(ValueError):
        ClinicalLLMRegressor(model.backbone)


@pytest.mark.parametrize("side", ["left", "right"])
def test_eos_reused_as_padding_never_selected(side):
    model = make_model(side=side)
    model.backbone.tokenizer.pad_token = model.backbone.tokenizer.eos_token
    model.llm.config.pad_token_id = model.backbone.tokenizer.pad_token_id
    tokens = ClinicalTokenizer(model.backbone)(["age 68", ""])
    assert (tokens.input_ids[tokens.attention_mask == 0] == model.backbone.tokenizer.eos_token_id).all()
    for row, position in enumerate(tokens.prediction_positions):
        assert tokens.attention_mask[row, position] == 1
        assert tokens.input_ids[row, position] != model.backbone.tokenizer.eos_token_id
    assert torch.isfinite(forward(model, tokens)).all()
