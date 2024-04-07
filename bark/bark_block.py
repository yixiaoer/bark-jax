from jax import Array
import jax.numpy as jnp
import jax.random as jrand
import torch
from transformers import BarkModel
from transformers.models.bark.modeling_bark import BarkBlock

from bark.array_conversion import jax2pt, pt2jax
from bark.attention import AttentionParams, convert_attention_params, forward_attention
from bark.layernorm import LayerNormParams, convert_layer_norm_params, forward_layer_norm
from bark.mlp_layer import MLPLayerParams, convert_mlp_layer_params, forward_mlp_layer

# TODO: eliminate this
d_model = 1024

BarkBlockParams = tuple[LayerNormParams, LayerNormParams, AttentionParams, MLPLayerParams]

def convert_bark_block_params(bark_block: BarkBlock) -> BarkBlockParams:
    return(
        convert_layer_norm_params(bark_block.layernorm_1),
        convert_layer_norm_params(bark_block.layernorm_2),
        convert_attention_params(bark_block.attn),
        convert_mlp_layer_params(bark_block.mlp),
    )

def forward_bark_block(params: BarkBlockParams, input_: Array, qk_mask: Array) -> Array:
    layernorm_1, layernorm_2, self_attn, mlp = params
    input_norm = forward_layer_norm(layernorm_1, input_)
    input_ += forward_attention(self_attn, input_norm, qk_mask)
    input_norm = forward_layer_norm(layernorm_2, input_)
    input_ += forward_mlp_layer(mlp, input_norm)
    return input_

def test_forward_bark_block(model: BarkModel) -> None:
    batch_size, seq_len = 4, 11
    bark_block_pt = model.semantic.layers[0]

    key = jrand.key(42)
    key, subkey = jrand.split(key)
    x = jrand.normal(subkey, (batch_size, seq_len, d_model))
    x_pt = jax2pt(x)

    attention_mask_pt = torch.tril(torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=model.device))
    attention_mask_pt_ = torch.where(attention_mask_pt, 0., -torch.inf)
    out_pt = bark_block_pt(x_pt, attention_mask=attention_mask_pt_)[0]

    bark_block_params = convert_bark_block_params(bark_block_pt)
    attention_mask_jax = pt2jax(attention_mask_pt)
    out = forward_bark_block(bark_block_params, x, attention_mask_jax)

    assert jnp.allclose(out, pt2jax(out_pt), atol=1e-5)
