from jax import Array
import jax.numpy as jnp
import jax.random as jrand
from torch.nn import Embedding as TorchEmbedding
from transformers import BarkModel

from bark.array_conversion import jax2pt, pt2jax

EmbeddingParams = Array

def convert_embedding_params(embedding: TorchEmbedding) -> EmbeddingParams:
    """
    Converts PyTorch embedding parameters to a EmbeddingParams compatible with JAX.

    Args:
        embedding (TorchEmbedding): The PyTorch embedding layer from which to extract the weights.

    Returns:
        EmbeddingParams: The embedding parameters extracted from the PyTorch layer and formatted for compatibility with JAX operations.
    """
    return pt2jax(embedding.weight.data)

def convert_back_embedding_params():
    pass

def forward_embedding(params: EmbeddingParams, input_ids: Array) -> Array:
    """
    Get the embedding with input IDS.

    Args:
        params (EmbeddingParams): The embedding parameters.
        input_ids (Array): An array of input IDS to look up the embedding.

    Returns:
        Array: The embedding Array of input IDS.
    """
    return params[input_ids]

def test_forward_embedding(model: BarkModel) -> None:
    """
    Tests the embedding parameters.

    Args:
        model (BarkModel): PyTorch Bark model to compare with this implementation.

    Returns:
        None.
    """
    batch_size, len_ = 4, 10
    embedding_pt = model.semantic.input_embeds_layer

    key = jrand.key(42)
    key, subkey = jrand.split(key)
    x = jrand.randint(subkey, (batch_size, len_), 0, 100)
    x_pt = jax2pt(x)

    out_pt = embedding_pt(x_pt)

    embedding_param = convert_embedding_params(embedding_pt)
    out = forward_embedding(embedding_param, x)

    assert jnp.allclose(out, pt2jax(out_pt), atol=1e-5)
