import math

import einops as op
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jrand
import torch
from transformers import BarkModel
from transformers.models.bark.modeling_bark import BarkSelfAttention
from bark.array_conversion import jax2pt, pt2jax

# TODO: eliminate this
d_model = 1024
n_heads = 16
d_k = d_v = 64

AttentionParams = tuple[Array, Array, Array, Array]

def convert_attention_params(self_attn: BarkSelfAttention) -> AttentionParams:
    q_proj = self_attn.att_proj.weight.data[: d_model, :]
    k_proj = self_attn.att_proj.weight.data[d_model : d_model * 2, :]
    v_proj = self_attn.att_proj.weight.data[d_model * 2 : d_model * 3, :]
    o_proj = self_attn.out_proj.weight.data

    q_proj_jax = pt2jax(q_proj.T).reshape(d_model, n_heads, d_k)
    k_proj_jax = pt2jax(k_proj.T).reshape(d_model, n_heads, d_k)
    v_proj_jax = pt2jax(v_proj.T).reshape(d_model, n_heads, d_v)
    o_proj_jax = pt2jax(o_proj.T).reshape(n_heads, d_v, d_model)

    return q_proj_jax, k_proj_jax, v_proj_jax, o_proj_jax

def forward_attention(params: AttentionParams, seq: Array, qk_mask: Array) -> Array:
    q_proj_jax, k_proj_jax, v_proj_jax, o_proj_jax = params

    q = op.einsum(seq, q_proj_jax, 'b s m, m h k -> b h s k')
    k = op.einsum(seq, k_proj_jax, 'b d m, m h k -> b h d k')
    v = op.einsum(seq, v_proj_jax, 'b d m, m h v -> b h d v')

    # Scaled Dot-Product Attention as 3.2.1 equation(1) in orginal Transformer paper
    qk = jnp.einsum('bhsk,bhdk->bhsd', q, k) / math.sqrt(d_k)

    qk = jax.nn.softmax(qk, where=qk_mask, initial=0.)
    qkv = jnp.einsum('bhsd,bhdv->bhsv', qk, v)
    out = jnp.einsum('bhsv,hvm->bsm', qkv, o_proj_jax)
    return out

def test_forward_attention(model: BarkModel) -> None:
    batch_size, seq_len = 1, 10

    self_attn_pt = model.semantic.layers[0].attn

    key = jrand.key(42)
    key, subkey = jrand.split(key)
    x = jrand.normal(subkey, (batch_size, seq_len, d_model))
    x_pt = jax2pt(x)

    attention_mask_pt = torch.tril(torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=model.device))
    attention_mask_pt_ = torch.where(attention_mask_pt, 0., -torch.inf)
    out_pt = self_attn_pt(x_pt, attention_mask=attention_mask_pt_)[0]

    attn_params = convert_attention_params(self_attn_pt)
    attention_mask_jax = pt2jax(attention_mask_pt)
    batch_size, seq_len, _ = x.shape
    out = forward_attention(attn_params, x, attention_mask_jax)

    assert jnp.allclose(out, pt2jax(out_pt), atol=1e-5)
