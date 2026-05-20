from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.activations import ACT2FN
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2LMHeadModel



class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * x.shape[-1] ** (-0.5)
        return self.scale * (x / (rms_x + self.eps))


class MyConv1D(nn.Module):
    def __init__(
        self,
        nf: int,
        nx: int,
        resid_gain: Optional[float] = None,
        skip_gain: Optional[float] = None,
        trainable_gains: bool = False,
        init_type: str = "normal",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.nf = nf
        self.bias = nn.Parameter(torch.zeros(nf), requires_grad=bias)
        if skip_gain is None:
            self.weight = nn.Parameter(torch.empty(nx, nf))
            if init_type == "orth":
                nn.init.orthogonal_(self.weight)
            elif init_type == "id":
                self.weight.data = torch.eye(nx)
            else:
                nn.init.normal_(self.weight, std=0.02)
            self.skip = False
        else:
            assert nx == nf
            self.resid_gain = nn.Parameter(
                torch.tensor([resid_gain]), requires_grad=trainable_gains
            )
            self.skip_gain = nn.Parameter(
                torch.tensor([skip_gain]), requires_grad=trainable_gains
            )
            self.weight = nn.Parameter(torch.zeros(nx, nx))
            if init_type == "orth":
                self.id = nn.init.orthogonal_(torch.empty(nx, nx))
            elif init_type == "id":
                self.id = torch.eye(nx)
            else:
                self.id = nn.init.normal_(torch.empty(nx, nx), std=1 / math.sqrt(nx))
            self.skip = True
            self.init_type = init_type

    def forward(self, x: Tensor) -> Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        if getattr(self, "skip", False):
            if self.resid_gain == 0 and self.init_type == "id":
                x = torch.add(self.bias, x * self.skip_gain)
            else:
                w = self.resid_gain * self.weight + self.skip_gain * self.id
                x = torch.addmm(self.bias, x.view(-1, x.size(-1)), w)
        else:
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 1e-2) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        return torch.where(input >= 0.0, input, input * self.negative_slope)


class myGPT2MLP(nn.Module):
    def __init__(self, intermediate_size: int, config: Any) -> None:
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = MyConv1D(intermediate_size, embed_dim, bias=False)
        self.c_proj = MyConv1D(embed_dim, intermediate_size, bias=False)
        if config.activation_function != "leaky_relu":
            self.act = ACT2FN[config.activation_function]
        else:
            self.act = LeakyReLU(negative_slope=config.lrelu_neg_slope)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)


class myGPT2Attention(nn.Module):
    def __init__(self, config: Any, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        max_positions = config.max_position_embeddings
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx

        self.qk_attn = MyConv1D(2 * self.embed_dim, self.embed_dim)
        self.split_size = self.embed_dim

        if config.value_skip_gain is not None or config.value_resid_gain != 0:
            self.v_attn = MyConv1D(
                self.embed_dim,
                self.embed_dim,
                resid_gain=config.value_resid_gain,
                skip_gain=config.value_skip_gain,
                trainable_gains=config.trainable_value_gains,
                init_type=config.val_init_type,
                bias=False,
            )
        else:
            self.v_attn = nn.Identity()

        proj_resid_gain = config.proj_resid_gain
        if config.proj_skip_gain is not None or proj_resid_gain != 0:
            self.c_proj = MyConv1D(
                self.embed_dim,
                self.embed_dim,
                resid_gain=proj_resid_gain,
                skip_gain=config.proj_skip_gain,
                trainable_gains=config.trainable_proj_gains,
                init_type=config.proj_init_type,
                bias=False,
            )
        else:
            self.c_proj = nn.Identity()

        if config.qk_norm_type == "rmsnorm":
            self.q_norm = RMSNorm(self.head_dim, eps=config.layer_norm_epsilon)
            self.k_norm = RMSNorm(self.head_dim, eps=config.layer_norm_epsilon)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.dot_norm = (
            nn.Identity()
            if config.dot_norm_type == "none"
            else (lambda x, sc=30.0: sc * torch.tanh(x.float() / sc)).__call__  # type: ignore
        )
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.attn_mat_resid_gain = nn.Parameter(
            config.attn_mat_resid_gain * torch.ones((1, self.num_heads, 1, 1)),
            requires_grad=config.trainable_attn_mat_gains,
        )
        self.attn_mat_skip_gain = nn.Parameter(
            config.attn_mat_skip_gain * torch.ones((1, self.num_heads, 1, 1)),
            requires_grad=config.trainable_attn_mat_gains,
        )
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )

    def _attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_weights = attn_weights / math.sqrt(value.size(-1))
        if self.scale_attn_by_inverse_layer_idx and self.layer_idx is not None:
            attn_weights = attn_weights / float(self.layer_idx + 1)
        attn_weights = self.dot_norm(attn_weights)

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        new_attn_weights = self.attn_mat_resid_gain * attn_weights.to(value.dtype)
        new_attn_weights = self.attn_dropout(new_attn_weights)
        if head_mask is not None:
            new_attn_weights = new_attn_weights * head_mask
        attn_output = torch.matmul(new_attn_weights, value)
        return attn_output, attn_weights

    def _split_heads(self, tensor: Tensor, num_heads: int, head_dim: int) -> Tensor:
        new_shape = tensor.size()[:-1] + (num_heads, head_dim)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor: Tensor, num_heads: int, head_dim: int) -> Tensor:
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * head_dim,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Tensor,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **_kwargs: Any,
    ) -> Tuple[Tensor, ...]:
        query, key = self.qk_attn(hidden_states).split(self.split_size, dim=2)
        value = self.v_attn(hidden_states)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        proj_output = self.c_proj(attn_output)
        proj_output = self.resid_dropout(proj_output)
        outputs = (proj_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class myGPT2Block(nn.Module):
    def __init__(self, config: Any, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.parallel_layers = config.parallel_layers
        self.norm_position = config.norm_position

        if config.norm_type == "ln":
            self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        elif config.norm_type == "rmsnorm":
            self.ln_1 = RMSNorm(hidden_size, eps=config.layer_norm_epsilon)
            self.ln_2 = RMSNorm(hidden_size, eps=config.layer_norm_epsilon)
        elif config.norm_type == "none":
            self.ln_1 = (
                nn.Identity()
                if layer_idx is not None and layer_idx > 0
                else nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            )
            self.ln_2 = nn.Identity()
        else:
            raise NotImplementedError(config.norm_type)

        self.attn = myGPT2Attention(config, layer_idx=layer_idx)
        self.mlp = myGPT2MLP(inner_dim, config)
        self.attn_block_resid_gain = nn.Parameter(
            torch.tensor([config.attn_block_resid_gain]),
            requires_grad=config.trainable_attn_block_gains,
        )
        self.attn_block_skip_gain = nn.Parameter(
            torch.tensor([config.attn_block_skip_gain]),
            requires_grad=config.trainable_attn_block_gains and not self.parallel_layers,
        )
        self.mlp_block_resid_gain = nn.Parameter(
            torch.tensor([config.mlp_block_resid_gain]),
            requires_grad=config.trainable_mlp_block_gains,
        )
        self.mlp_block_skip_gain = nn.Parameter(
            torch.tensor([config.mlp_block_skip_gain]),
            requires_grad=config.trainable_mlp_block_gains and not self.parallel_layers,
        )
        self.add_attn_skip = config.attn_block_skip_gain != 0
        self.add_mlp_skip = config.mlp_block_skip_gain != 0

    def forward(
        self,
        hidden_states: Tensor,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **_kwargs: Any,
    ) -> Tuple[Tensor, ...]:
        if self.norm_position == "post":
            hidden_states = self.ln_1(hidden_states)
        skip_branch = hidden_states
        if self.norm_position == "pre":
            hidden_states = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        if self.parallel_layers:
            feed_forward_hidden_states = self.mlp(hidden_states)
            hidden_states = (
                self.mlp_block_resid_gain * feed_forward_hidden_states
                + self.attn_block_resid_gain * attn_output
            )
            if self.add_mlp_skip:
                hidden_states = hidden_states + self.mlp_block_skip_gain * skip_branch
        else:
            hidden_states = self.attn_block_resid_gain * attn_output
            if self.add_attn_skip:
                hidden_states = hidden_states + self.attn_block_skip_gain * skip_branch

            if self.norm_position == "post":
                hidden_states = self.ln_2(hidden_states)
            skip_branch = hidden_states
            if self.norm_position == "pre":
                hidden_states = self.ln_2(hidden_states)

            feed_forward_hidden_states = self.mlp(hidden_states)
            hidden_states = self.mlp_block_resid_gain * feed_forward_hidden_states
            if self.add_mlp_skip:
                hidden_states = hidden_states + self.mlp_block_skip_gain * skip_branch

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs


def _op_block_config(config: Any) -> None:
    n_layer = int(config.n_layer)
    gain = 1.0 / math.sqrt(n_layer)
    config.norm_type = "none"
    config.norm_position = "pre"
    config.qk_norm_type = "rmsnorm"
    config.dot_norm_type = "none"
    config.parallel_layers = False
    config.attn_block_resid_gain = gain
    config.mlp_block_resid_gain = gain
    config.attn_block_skip_gain = 1.0
    config.mlp_block_skip_gain = 1.0
    config.attn_mat_resid_gain = 1.0
    config.attn_mat_skip_gain = 0.0
    config.value_resid_gain = 1.0
    config.value_skip_gain = None
    config.proj_resid_gain = 1.0
    config.proj_skip_gain = None
    config.val_init_type = "normal"
    config.proj_init_type = "normal"
    config.activation_function = "leaky_relu"
    config.lrelu_neg_slope = 0.0
    config.trainable_attn_block_gains = False
    config.trainable_mlp_block_gains = False
    config.trainable_attn_mat_gains = False
    config.trainable_value_gains = False
    config.trainable_proj_gains = False
    config.centre_attn = False
    config.centre_attn_gain = 1.0


def apply_op_blocks(model: GPT2LMHeadModel) -> GPT2LMHeadModel:
    # adapted from https://github.com/bobby-he/simplified_transformers
    _op_block_config(model.config)
    new_blocks = []
    for i, _ in enumerate(model.transformer.h):
        new_blocks.append(myGPT2Block(model.config, layer_idx=i))
    model.transformer.h = nn.ModuleList(new_blocks)
    return model



def _resolve_hidden_states(hidden_states: Any, kwargs: dict[str, Any]) -> Tensor:
    if hidden_states is None and kwargs:
        hidden_states = next(iter(kwargs.values()))
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]
    return hidden_states


def _gpt2_split_heads(tensor: Tensor, num_heads: int, head_dim: int) -> Tensor:
    return tensor.view(*tensor.shape[:-1], num_heads, head_dim).transpose(1, 2)


def _gpt2_merge_heads(tensor: Tensor) -> Tensor:
    return tensor.reshape(*tensor.shape[:-2], -1).contiguous()


def _gpt2_project_qkv(attn: GPT2Attention, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    query, key, value = attn.c_attn(hidden_states).split(attn.split_size, dim=2)
    nh, hd = attn.num_heads, attn.head_dim
    return (
        _gpt2_split_heads(query, nh, hd),
        _gpt2_split_heads(key, nh, hd),
        _gpt2_split_heads(value, nh, hd),
    )


def _gpt2_eager_attention(
    attn: GPT2Attention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
) -> tuple[Tensor, Tensor]:
    scaling = getattr(attn, "scaling", attn.head_dim**-0.5)
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = attn.attn_dropout(attn_weights)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights


def _gpt2_finish_attention(attn: GPT2Attention, attn_output: Tensor) -> Tensor:
    attn_output = _gpt2_merge_heads(attn_output)
    attn_output = attn.c_proj(attn_output)
    return attn.resid_dropout(attn_output)


class GPT2AttentionWithKVBias(GPT2Attention):
    def __init__(self, config: Any, is_cross_attention: bool = False, layer_idx: Optional[int] = None):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        head_dim = self.head_dim
        self.k_bias = nn.Parameter(torch.zeros(1, self.num_heads, 1, head_dim))
        self.v_bias = nn.Parameter(torch.zeros(1, self.num_heads, 1, head_dim))
        nn.init.normal_(self.k_bias, mean=0.0, std=0.02)
        nn.init.normal_(self.v_bias, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states: Any = None,
        past_key_values: Any = None,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[Tensor, Optional[Tensor]]:
        del past_key_values, encoder_hidden_states, encoder_attention_mask, use_cache
        hidden_states = _resolve_hidden_states(hidden_states, kwargs)

        query, key, value = _gpt2_project_qkv(self, hidden_states)
        batch_size = hidden_states.size(0)
        key = torch.cat((self.k_bias.expand(batch_size, -1, -1, -1), key), dim=2)
        value = torch.cat((self.v_bias.expand(batch_size, -1, -1, -1), value), dim=2)

        query_len = query.size(-2)
        seq_len = key.size(-2)
        causal = torch.ones(
            query_len, seq_len, dtype=torch.bool, device=hidden_states.device
        ).tril(diagonal=seq_len - query_len)
        mask_value = torch.finfo(query.dtype).min
        attn_mask = torch.zeros(1, 1, query_len, seq_len, device=query.device, dtype=query.dtype)
        attn_mask = attn_mask.masked_fill(~causal.view(1, 1, query_len, seq_len), mask_value)
        if attention_mask is not None:
            attn_mask = attn_mask + attention_mask

        attn_output, attn_weights = _gpt2_eager_attention(self, query, key, value, attn_mask)
        attn_output = _gpt2_finish_attention(self, attn_output)
        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None


class GPT2AttentionWithKVBiasAndCAScale(GPT2Attention):
    """Systematic-outliers attention bias plus context-aware scaling."""

    def __init__(self, config: Any, is_cross_attention: bool = False, layer_idx: Optional[int] = None):
        GPT2Attention.__init__(self, config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        head_dim = self.head_dim
        self.k_bias = nn.Parameter(torch.zeros(1, self.num_heads, 1, head_dim))
        self.v_bias = nn.Parameter(torch.zeros(1, self.num_heads, 1, head_dim))
        nn.init.normal_(self.k_bias, mean=0.0, std=0.02)
        nn.init.normal_(self.v_bias, mean=0.0, std=0.02)
        self.context_scale = nn.Sequential(
            nn.Linear(config.hidden_size, config.num_attention_heads),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: Any = None,
        past_key_values: Any = None,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[Tensor, Optional[Tensor]]:
        del past_key_values, encoder_hidden_states, encoder_attention_mask, use_cache
        hidden_states = _resolve_hidden_states(hidden_states, kwargs)

        query, key, value = _gpt2_project_qkv(self, hidden_states)
        batch_size = hidden_states.size(0)
        key = torch.cat((self.k_bias.expand(batch_size, -1, -1, -1), key), dim=2)
        value = torch.cat((self.v_bias.expand(batch_size, -1, -1, -1), value), dim=2)

        query_len = query.size(-2)
        seq_len = key.size(-2)
        causal = torch.ones(
            query_len, seq_len, dtype=torch.bool, device=hidden_states.device
        ).tril(diagonal=seq_len - query_len)
        mask_value = torch.finfo(query.dtype).min
        attn_mask = torch.zeros(1, 1, query_len, seq_len, device=query.device, dtype=query.dtype)
        attn_mask = attn_mask.masked_fill(~causal.view(1, 1, query_len, seq_len), mask_value)
        if attention_mask is not None:
            attn_mask = attn_mask + attention_mask

        attn_output, attn_weights = _gpt2_eager_attention(self, query, key, value, attn_mask)
        s_c = 2.0 * self.context_scale(hidden_states)
        attn_output = attn_output * s_c.permute(0, 2, 1).unsqueeze(-1)
        attn_output = _gpt2_finish_attention(self, attn_output)
        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None


class GPT2AttentionWithCAScale(GPT2Attention):
    """Context-aware scaling factor S(c) on attention output (systematic-outliers)."""

    def __init__(self, config: Any, is_cross_attention: bool = False, layer_idx: Optional[int] = None):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.context_scale = nn.Sequential(
            nn.Linear(config.hidden_size, config.num_attention_heads),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: Any = None,
        past_key_values: Any = None,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[Tensor, Optional[Tensor]]:
        del past_key_values, encoder_hidden_states, encoder_attention_mask, use_cache
        hidden_states = _resolve_hidden_states(hidden_states, kwargs)

        query, key, value = _gpt2_project_qkv(self, hidden_states)
        attn_output, attn_weights = _gpt2_eager_attention(self, query, key, value, attention_mask)
        s_c = 2.0 * self.context_scale(hidden_states)
        attn_output = attn_output * s_c.permute(0, 2, 1).unsqueeze(-1)
        attn_output = _gpt2_finish_attention(self, attn_output)
        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None


def set_attention_linear_bias(model: GPT2LMHeadModel, enabled: bool) -> None:
    for block in model.transformer.h:
        if not isinstance(block, GPT2Block):
            continue
        attn = block.attn
        if not isinstance(attn, GPT2Attention):
            continue
        for name in ("c_attn", "c_proj"):
            conv = getattr(attn, name)
            if conv.bias is None:
                conv.bias = nn.Parameter(
                    torch.zeros(conv.weight.shape[0], device=conv.weight.device, dtype=conv.weight.dtype)
                )
            if enabled:
                conv.bias.requires_grad_(True)
            else:
                with torch.no_grad():
                    conv.bias.zero_()
                conv.bias.requires_grad_(False)


def _copy_attn_weights(dst: GPT2Attention, src: GPT2Attention) -> None:
    dst.c_attn.weight.data.copy_(src.c_attn.weight.data)
    dst.c_proj.weight.data.copy_(src.c_proj.weight.data)
    if src.c_attn.bias is not None and dst.c_attn.bias is not None:
        dst.c_attn.bias.data.copy_(src.c_attn.bias.data)
    if src.c_proj.bias is not None and dst.c_proj.bias is not None:
        dst.c_proj.bias.data.copy_(src.c_proj.bias.data)


def _replace_attention(
    model: GPT2LMHeadModel,
    attn_cls: type,
    layer_idx: int,
    old_attn: GPT2Attention,
) -> GPT2Attention:
    new_attn = attn_cls(model.config, layer_idx=layer_idx)
    _copy_attn_weights(new_attn, old_attn)
    return new_attn


def apply_attention_bias(model: GPT2LMHeadModel) -> GPT2LMHeadModel:
    for i, block in enumerate(model.transformer.h):
        if not isinstance(block, GPT2Block):
            continue
        old_attn = block.attn
        if not isinstance(old_attn, GPT2Attention):
            continue
        block.attn = _replace_attention(model, GPT2AttentionWithKVBias, i, old_attn)
    return model


def apply_context_aware_scaling(model: GPT2LMHeadModel) -> GPT2LMHeadModel:
    # adapted from https://github.com/an-yongqi/systematic-outliers
    for i, block in enumerate(model.transformer.h):
        if not isinstance(block, GPT2Block):
            continue
        old_attn = block.attn
        if not isinstance(old_attn, GPT2Attention):
            continue
        if isinstance(old_attn, GPT2AttentionWithKVBias):
            new_attn = GPT2AttentionWithKVBiasAndCAScale(model.config, layer_idx=i)
            _copy_attn_weights(new_attn, old_attn)
            new_attn.k_bias.data.copy_(old_attn.k_bias.data)
            new_attn.v_bias.data.copy_(old_attn.v_bias.data)
            block.attn = new_attn
        else:
            block.attn = _replace_attention(model, GPT2AttentionWithCAScale, i, old_attn)
    return model



def fake_quantize_affine_ste(x: Tensor, num_bits: int = 8) -> Tensor:
    if not x.is_floating_point():
        return x
    qmin = 0
    qmax = (1 << num_bits) - 1
    x_min = torch.amin(x)
    x_max = torch.amax(x)
    if torch.isclose(x_max, x_min):
        return x
    scale = (x_max - x_min) / float(qmax - qmin)
    zero_point = torch.clamp(qmin - torch.round(x_min / scale), qmin, qmax)
    q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
    dequant = (q - zero_point) * scale
    return x + (dequant - x).detach()


_qat_handles: list[Any] = []


def enable_qat_training(model: nn.Module) -> None:
    disable_qat_training()

    def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Tensor) -> Tensor:
        if isinstance(output, torch.Tensor) and output.is_floating_point():
            return fake_quantize_affine_ste(output)
        return output

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)) or module.__class__.__name__ == "Conv1D":
            _qat_handles.append(module.register_forward_hook(_hook))


def disable_qat_training() -> None:
    for handle in _qat_handles:
        handle.remove()
    _qat_handles.clear()
