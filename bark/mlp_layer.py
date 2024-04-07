import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jrand
from transformers import BarkModel
from transformers.models.bark.modeling_bark import BarkMLP

from bark.array_conversion import jax2pt, pt2jax
from bark.dropout import forward_dropout

# TODO: eliminate this
d_model = 1024

MLPLayerParams = tuple[Array, Array]

def convert_mlp_layer_params(mlp_layer: BarkMLP) -> MLPLayerParams:
    in_proj = pt2jax(mlp_layer.in_proj.weight.data.T)
    out_proj = pt2jax(mlp_layer.out_proj.weight.data.T)
    return in_proj, out_proj

def forward_mlp_layer(params: MLPLayerParams, input_: Array) -> Array:
    in_proj, out_proj = params
    out = jax.nn.gelu(input_ @ in_proj, approximate=False) @ out_proj
    out = forward_dropout(out)
    return out

def test_forward_mlp_layer(model: BarkModel) -> None:
    batch_size, seq_len = 4, 10
    mlp_pt = model.semantic.layers[0].mlp

    key = jrand.key(42)
    key, subkey = jrand.split(key)
    x = jrand.normal(subkey, (batch_size, seq_len, d_model))
    x_pt = jax2pt(x)

    out_pt = mlp_pt(x_pt)

    mlp_params = convert_mlp_layer_params(mlp_pt)
    out = forward_mlp_layer(mlp_params, x)

    assert jnp.allclose(out, pt2jax(out_pt), atol=1e-5)
