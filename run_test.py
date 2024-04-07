import jax; jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

from transformers import BarkModel

from bark.attention import test_forward_attention
from bark.embedding import test_forward_embedding
from bark.layernorm import test_forward_layer_norm

def main():
    model = BarkModel.from_pretrained("suno/bark")
    test_forward_attention(model)
    test_forward_embedding(model)
    test_forward_layer_norm(model)
    print('✅ All tests passed!')

if __name__ == '__main__':
    main()
