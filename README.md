# 🐶 Bark JAX
This project is the JAX implementation of [Bark](https://github.com/suno-ai/bark).

It is supported by Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).

## Roadmap

- [ ] Model architecture
    - [x] EnCodec (encodec_24khz), implemented in [encodec jax](https://github.com/yixiaoer/encodec-jax)
    - [ ] 3 transformer models
        - [ ] Text to semantic tokens
        - [ ] Semantic to coarse tokens
        - [ ] Coarse to fine tokens

## Install

This project requires Python 3.12, JAX 0.4.25.

Create venv:

```sh
python3.12 -m venv venv
```

install dependencies:

TPU VM:

```sh
. venv/bin/activate
pip install -U pip
pip install -U wheel
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install git+https://github.com/huggingface/transformers
```

## Model Architecture

Bark is a series of three transformer models and one EnCodec model, to turn text into audio.

### Transformer model: Text to semantic tokens

```
BarkSemanticModel(
  (input_embeds_layer): Embedding(129600, 1024)
  (position_embeds_layer): Embedding(1024, 1024)
  (drop): Dropout(p=0.0, inplace=False)
  (layers): ModuleList(
    (0-23): 24 x BarkBlock(
      (layernorm_1): BarkLayerNorm()
      (layernorm_2): BarkLayerNorm()
      (attn): BarkSelfAttention(
        (attn_dropout): Dropout(p=0.0, inplace=False)
        (resid_dropout): Dropout(p=0.0, inplace=False)
        (att_proj): Linear(in_features=1024, out_features=3072, bias=False)
        (out_proj): Linear(in_features=1024, out_features=1024, bias=False)
      )
      (mlp): BarkMLP(
        (in_proj): Linear(in_features=1024, out_features=4096, bias=False)
        (out_proj): Linear(in_features=4096, out_features=1024, bias=False)
        (dropout): Dropout(p=0.0, inplace=False)
        (gelu): GELU(approximate='none')
      )
    )
  )
  (layernorm_final): BarkLayerNorm()
  (lm_head): Linear(in_features=1024, out_features=10048, bias=False)
)
```

### Transformer model: Semantic to coarse tokens

```
BarkCoarseModel(
  (input_embeds_layer): Embedding(12096, 1024)
  (position_embeds_layer): Embedding(1024, 1024)
  (drop): Dropout(p=0.0, inplace=False)
  (layers): ModuleList(
    (0-23): 24 x BarkBlock(
      (layernorm_1): BarkLayerNorm()
      (layernorm_2): BarkLayerNorm()
      (attn): BarkSelfAttention(
        (attn_dropout): Dropout(p=0.0, inplace=False)
        (resid_dropout): Dropout(p=0.0, inplace=False)
        (att_proj): Linear(in_features=1024, out_features=3072, bias=False)
        (out_proj): Linear(in_features=1024, out_features=1024, bias=False)
      )
      (mlp): BarkMLP(
        (in_proj): Linear(in_features=1024, out_features=4096, bias=False)
        (out_proj): Linear(in_features=4096, out_features=1024, bias=False)
        (dropout): Dropout(p=0.0, inplace=False)
        (gelu): GELU(approximate='none')
      )
    )
  )
  (layernorm_final): BarkLayerNorm()
  (lm_head): Linear(in_features=1024, out_features=12096, bias=False)
)
```

### Transformer model: Coarse to fine tokens

```
BarkFineModel(
  (input_embeds_layers): ModuleList(
    (0-7): 8 x Embedding(1056, 1024)
  )
  (position_embeds_layer): Embedding(1024, 1024)
  (drop): Dropout(p=0.0, inplace=False)
  (layers): ModuleList(
    (0-23): 24 x BarkBlock(
      (layernorm_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (layernorm_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (attn): BarkSelfAttention(
        (attn_dropout): Dropout(p=0.0, inplace=False)
        (resid_dropout): Dropout(p=0.0, inplace=False)
        (att_proj): Linear(in_features=1024, out_features=3072, bias=False)
        (out_proj): Linear(in_features=1024, out_features=1024, bias=False)
      )
      (mlp): BarkMLP(
        (in_proj): Linear(in_features=1024, out_features=4096, bias=False)
        (out_proj): Linear(in_features=4096, out_features=1024, bias=False)
        (dropout): Dropout(p=0.0, inplace=False)
        (gelu): GELU(approximate='none')
      )
    )
  )
  (layernorm_final): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  (lm_heads): ModuleList(
    (0-6): 7 x Linear(in_features=1024, out_features=1056, bias=False)
  )
)
```

### EnCodec

For detailed information, go to [EnCodec JAX implementation](https://github.com/yixiaoer/encodec-jax), [EnCodec official](https://github.com/facebookresearch/encodec), and paper [High Fidelity Neural Audio Compression](https://arxiv.org/pdf/2210.13438.pdf).


