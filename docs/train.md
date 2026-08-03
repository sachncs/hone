# Training

## Subcommands

```text
hone train code     train a coding adapter
hone train swe      train an SWE adapter
hone train all      full sequence across every dataset
```

The `code` and `swe` subcommands read a YAML config (default
`configs/code.yaml` and `configs/swe.yaml`) and invoke
`python -m hone.run --config <path>` with `HONE_DEVICE` set
in the subprocess environment.

## Configs

The YAML config keys are the upstream `mlx_lm.lora` contract.
`hone` does not rename them.

```yaml
model: openbmb/MiniCPM5-1B
train: true
fine_tune_type: lora
data: data/processed/code
seed: 42
num_layers: 12
batch_size: 1
grad_accumulation_steps: 8
iters: 1200
val_batches: 25
learning_rate: 0.00001
adapter_path: artifacts/code-lora
save_every: 200
max_seq_length: 3072
grad_checkpoint: true
mask_prompt: true
lora_parameters:
  keys: [self_attn.q_proj, self_attn.v_proj]
  rank: 8
  scale: 16.0
  dropout: 0.05
```

`hone.config.validate` checks that `model`, `train`, and `data`
are present before training starts.

## Backend

```bash
hone train code --backend mlx   # default
hone train code --backend cuda  # Unsloth fallback
```

## Tuning

```bash
hone tune run \
    --config configs/code.yaml \
    --space configs/tune-code.yaml \
    --output artifacts/tuning/code
```

Each trial writes its own `config.yaml`, runs `python -m hone.run`,
and reports its validation loss. `results.json` is the full set;
`best.json` is selected by the objective.
