from jax import Array
import jax.random as jrand

def forward_dropout(input_: Array, p: float = 0., key: Array | None = None) -> Array:
    if p == 0. or key is None:
        return input_
    keep_p = 1. - p
    return input_ * jrand.bernoulli(key, p=keep_p, shape=input_.shape) / keep_p
