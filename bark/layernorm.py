from jax import Array
import jax.numpy as jnp
import jax.random as jrand
from transformers import BarkModel
from transformers.models.bark.modeling_bark import BarkLayerNorm
from bark.array_conversion import jax2pt, pt2jax

LayerNormParams = tuple[Array, None | Array]

def convert_layer_norm_params(layernorm: BarkLayerNorm) -> LayerNormParams:
    return (
        pt2jax(layernorm.weight),
        None if layernorm.bias is None else pt2jax(layernorm.bias),
    )

def forward_layer_norm(params: LayerNormParams, input_: Array, eps: float = 1e-5) -> Array:
    weight, bias = params
    normalized_input = (input_ - jnp.mean(input_, axis=-1, keepdims=True)) / jnp.sqrt(jnp.var(input_, axis=-1, keepdims=True) + eps)
    out = weight * normalized_input
    if bias is not None:
        out += bias
    return out

def test_forward_layer_norm(model: BarkModel) -> None:
    batch_size, hidden_size = 4, 1024
    layernorm_pt = model.semantic.layers[0].layernorm_1

    key = jrand.key(42)
    key, subkey = jrand.split(key)
    x = jrand.normal(subkey, (batch_size, hidden_size))
    x_pt = jax2pt(x)

    out_pt = layernorm_pt(x_pt)

    layernorm_param = convert_layer_norm_params(layernorm_pt)
    out = forward_layer_norm(layernorm_param, x)

    assert jnp.allclose(out, pt2jax(out_pt), atol=1e-5)
