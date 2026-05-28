from __future__ import annotations

from acta.target_layer_presets import resolve_target_layer_preset

GPT2: list[str] = [
    "transformer.h.*.attn.c_attn",
    "transformer.h.*.attn.c_proj",
    "transformer.h.*.mlp.c_fc",
    "transformer.h.*.mlp.c_proj",
]

ALBERT_BASE_V2: list[str] = [
    "albert.encoder.albert_layer_groups.*.albert_layers.*.attention.query",
    "albert.encoder.albert_layer_groups.*.albert_layers.*.attention.key",
    "albert.encoder.albert_layer_groups.*.albert_layers.*.attention.value",
    "albert.encoder.albert_layer_groups.*.albert_layers.*.attention.dense",
    "albert.encoder.albert_layer_groups.*.albert_layers.*.ffn",
    "albert.encoder.albert_layer_groups.*.albert_layers.*.ffn_output",
]

DEBERTA_V3: list[str] = [
    "deberta.encoder.layer.*.attention.self.query_proj",
    "deberta.encoder.layer.*.attention.self.key_proj",
    "deberta.encoder.layer.*.attention.self.value_proj",
    "deberta.encoder.layer.*.attention.self.output_proj",
    "deberta.encoder.layer.*.intermediate.dense",
    "deberta.encoder.layer.*.output.dense",
]

BLOOM_560M: list[str] = [
    "transformer.h.*.self_attention.query_key_value",
    "transformer.h.*.self_attention.dense",
    "transformer.h.*.mlp.dense_h_to_4h",
    "transformer.h.*.mlp.dense_4h_to_h",
]

BART_BASE: list[str] = [
    "model.encoder.layers.*.self_attn.q_proj",
    "model.encoder.layers.*.self_attn.k_proj",
    "model.encoder.layers.*.self_attn.v_proj",
    "model.encoder.layers.*.self_attn.out_proj",
    "model.encoder.layers.*.fc1",
    "model.encoder.layers.*.fc2",
    "model.decoder.layers.*.self_attn.q_proj",
    "model.decoder.layers.*.self_attn.k_proj",
    "model.decoder.layers.*.self_attn.v_proj",
    "model.decoder.layers.*.self_attn.out_proj",
    "model.decoder.layers.*.fc1",
    "model.decoder.layers.*.fc2",
]

T5_BASE: list[str] = [
    "encoder.block.*.layer.0.SelfAttention.q",
    "encoder.block.*.layer.0.SelfAttention.k",
    "encoder.block.*.layer.0.SelfAttention.v",
    "encoder.block.*.layer.0.SelfAttention.o",
    "encoder.block.*.layer.1.DenseReluDense.wi",
    "encoder.block.*.layer.1.DenseReluDense.wo",
]

VIT_DEIT_LAYERS: list[str] = [
    "vit.layers.*.attention.q_proj",
    "vit.layers.*.attention.k_proj",
    "vit.layers.*.attention.v_proj",
    "vit.layers.*.attention.o_proj",
    "vit.layers.*.mlp.fc1",
    "vit.layers.*.mlp.fc2",
    "deit.encoder.layer.*.attention.attention.query",
    "deit.encoder.layer.*.attention.attention.key",
    "deit.encoder.layer.*.attention.attention.value",
    "deit.encoder.layer.*.attention.output.dense",
    "deit.encoder.layer.*.intermediate.dense",
    "deit.encoder.layer.*.output.dense",
    "vit.encoder.layer.*.attention.attention.query",
    "vit.encoder.layer.*.attention.attention.key",
    "vit.encoder.layer.*.attention.attention.value",
    "vit.encoder.layer.*.attention.output.dense",
    "vit.encoder.layer.*.intermediate.dense",
    "vit.encoder.layer.*.output.dense",
]

SWIN_TINY: list[str] = [
    "swin.encoder.layers.*.blocks.*.attention.q_proj",
    "swin.encoder.layers.*.blocks.*.attention.k_proj",
    "swin.encoder.layers.*.blocks.*.attention.v_proj",
    "swin.encoder.layers.*.blocks.*.attention.o_proj",
    "swin.encoder.layers.*.blocks.*.mlp.fc1",
    "swin.encoder.layers.*.blocks.*.mlp.fc2",
]

IMAGEGPT_SMALL: list[str] = [
    "transformer.h.*.attn.c_attn",
    "transformer.h.*.attn.c_proj",
    "transformer.h.*.mlp.c_fc",
    "transformer.h.*.mlp.c_proj",
]

BLIP_CAPTIONING: list[str] = [
    "vision_model.encoder.layers.*.self_attn.qkv",
    "vision_model.encoder.layers.*.self_attn.projection",
    "vision_model.encoder.layers.*.mlp.fc1",
    "vision_model.encoder.layers.*.mlp.fc2",
    "text_decoder.bert.encoder.layer.*.attention.self.query",
    "text_decoder.bert.encoder.layer.*.attention.self.key",
    "text_decoder.bert.encoder.layer.*.attention.self.value",
    "text_decoder.bert.encoder.layer.*.attention.output.dense",
    "text_decoder.bert.encoder.layer.*.crossattention.self.query",
    "text_decoder.bert.encoder.layer.*.crossattention.self.key",
    "text_decoder.bert.encoder.layer.*.crossattention.self.value",
    "text_decoder.bert.encoder.layer.*.crossattention.output.dense",
    "text_decoder.bert.encoder.layer.*.intermediate.dense",
    "text_decoder.bert.encoder.layer.*.output.dense",
]

WHISPER_ENCODER: list[str] = [
    "model.encoder.layers.*.self_attn.q_proj",
    "model.encoder.layers.*.self_attn.k_proj",
    "model.encoder.layers.*.self_attn.v_proj",
    "model.encoder.layers.*.self_attn.out_proj",
    "model.encoder.layers.*.fc1",
    "model.encoder.layers.*.fc2",
    "encoder.layers.*.self_attn.q_proj",
    "encoder.layers.*.self_attn.k_proj",
    "encoder.layers.*.self_attn.v_proj",
    "encoder.layers.*.self_attn.out_proj",
    "encoder.layers.*.fc1",
    "encoder.layers.*.fc2",
]

HUBERT_BASE_LS960: list[str] = [
    "hubert.encoder.layers.*.attention.q_proj",
    "encoder.layers.*.attention.q_proj",
    "hubert.encoder.layers.*.attention.k_proj",
    "encoder.layers.*.attention.k_proj",
    "hubert.encoder.layers.*.attention.v_proj",
    "encoder.layers.*.attention.v_proj",
    "hubert.encoder.layers.*.attention.out_proj",
    "encoder.layers.*.attention.out_proj",
    "hubert.encoder.layers.*.feed_forward.intermediate_dense",
    "encoder.layers.*.feed_forward.intermediate_dense",
    "hubert.encoder.layers.*.feed_forward.output_dense",
    "encoder.layers.*.feed_forward.output_dense",
]

UNISPEECH_SAT_BASE: list[str] = [
    "unispeech_sat.encoder.layers.*.attention.q_proj",
    "encoder.layers.*.attention.q_proj",
    "unispeech_sat.encoder.layers.*.attention.k_proj",
    "encoder.layers.*.attention.k_proj",
    "unispeech_sat.encoder.layers.*.attention.v_proj",
    "encoder.layers.*.attention.v_proj",
    "unispeech_sat.encoder.layers.*.attention.out_proj",
    "encoder.layers.*.attention.out_proj",
    "unispeech_sat.encoder.layers.*.feed_forward.intermediate_dense",
    "encoder.layers.*.feed_forward.intermediate_dense",
    "unispeech_sat.encoder.layers.*.feed_forward.output_dense",
    "encoder.layers.*.feed_forward.output_dense",
]

_MODEL_MAP: dict[str, list[str]] = {
    "openai-community/gpt2": GPT2,
    "gpt2": GPT2,
    "albert/albert-base-v2": ALBERT_BASE_V2,
    "microsoft/deberta-v3-base": DEBERTA_V3,
    "bigscience/bloom-560m": BLOOM_560M,
    "facebook/bart-base": BART_BASE,
    "t5-base": T5_BASE,
    "openai/whisper-tiny": WHISPER_ENCODER,
    "openai/whisper-base": WHISPER_ENCODER,
    "facebook/hubert-base-ls960": HUBERT_BASE_LS960,
    "microsoft/unispeech-sat-base": UNISPEECH_SAT_BASE,
    "facebook/deit-tiny-patch16-224": VIT_DEIT_LAYERS,
    "google/vit-base-patch16-224": VIT_DEIT_LAYERS,
    "microsoft/swin-tiny-patch4-window7-224": SWIN_TINY,
    "openai/imagegpt-small": IMAGEGPT_SMALL,
    "Salesforce/blip-image-captioning-base": BLIP_CAPTIONING,
}


def target_layers_for_model(model_id: str) -> list[str]:
    """Return hook patterns for a Hugging Face model id."""
    mid = (model_id or "").strip()
    if mid in _MODEL_MAP:
        return list(_MODEL_MAP[mid])
    for candidate in (
        mid,
        f"distilbert/{mid}",
        f"openai-community/{mid}",
        f"google/{mid}",
        f"facebook/{mid}",
        f"microsoft/{mid}",
        f"albert/{mid}",
    ):
        preset = resolve_target_layer_preset(candidate)
        if preset:
            return list(preset)
    mid_low = mid.lower()
    for key, patterns in _MODEL_MAP.items():
        if key.lower() in mid_low or mid_low.endswith(key.lower().split("/")[-1]):
            return list(patterns)
    if "albert" in mid_low:
        return list(ALBERT_BASE_V2)
    if "deberta" in mid_low:
        return list(DEBERTA_V3)
    if "bloom" in mid_low:
        return list(BLOOM_560M)
    if "bart" in mid_low:
        return list(BART_BASE)
    if "t5" in mid_low:
        return list(T5_BASE)
    if "gpt2" in mid_low or "gpt-2" in mid_low:
        return list(GPT2)
    if "whisper" in mid_low:
        return list(WHISPER_ENCODER)
    if "hubert" in mid_low:
        return list(HUBERT_BASE_LS960)
    if "unispeech" in mid_low:
        return list(UNISPEECH_SAT_BASE)
    if "swin" in mid_low:
        return list(SWIN_TINY)
    if "imagegpt" in mid_low:
        return list(IMAGEGPT_SMALL)
    if "blip" in mid_low:
        return list(BLIP_CAPTIONING)
    if "vit" in mid_low or "deit" in mid_low:
        return list(VIT_DEIT_LAYERS)
    raise ValueError(
        f"No target_layers preset for {model_id!r}. "
        "Add patterns to examples/target_layers.py."
    )
