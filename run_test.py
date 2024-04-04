import jax; jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

from transformers import BarkModel

from bark.embedding import test_forward_embedding

def main():
    model = BarkModel.from_pretrained("suno/bark")
    test_forward_embedding(model)
    print('✅ All tests passed!')

if __name__ == '__main__':
    main()
