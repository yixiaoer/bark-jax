import jax; jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

from transformers import BarkModel

from bark.attention import test_forward_attention
from bark.bark_block import test_forward_bark_block
from bark.embedding import test_forward_embedding
from bark.layernorm import test_forward_layer_norm
from bark.mlp_layer import test_forward_mlp_layer

def main():
    model = BarkModel.from_pretrained("suno/bark")
    test_forward_attention(model)
    test_forward_bark_block(model)
    test_forward_embedding(model)
    test_forward_layer_norm(model)
    test_forward_mlp_layer(model)
    print('✅ All tests passed!')

if __name__ == '__main__':
    main()
