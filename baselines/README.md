# Baselines

Each baseline lives in its own folder, trains on the demos recorded with `make demos <task>`, saves under
`outputs/baselines/<baseline>/`, logs to Weights & Biases, and is evaluated in ManiSkill with the shared
`rollout.py` (success rate, each failure flag's rate, videos) on reset seeds the demos never used.
Run `wandb login` once per machine.
Every method is trained with 3 seeds (`TRAIN_SEED=42`, `43`, `44`); list each seed's checkpoint under the method's
`seeds` in `config/eval.yaml`, and `make eval` reports each seed plus the mean ± std across seeds (`aggregate`).

## Diffusion Policy (`diffusion_policy/`)

CNN U-Net Diffusion Policy (Chi et al., RSS 2023), ported from
[real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) (MIT, see `diffusion_policy/LICENSE`).
Observations are the last 2 frames of both cameras plus 7 arm joints and the finger opening; the policy predicts
16 future actions (next 7 joint targets + gripper command) and executes 8 with the `pd_joint_pos` controller.

| `encoder` | Visual backbone | Initialization |
|---|---|---|
| `scratch` | ResNet-18, GroupNorm, 128 px | random, trained end to end |
| `imagenet` | ResNet-18, BatchNorm, 128 px | ImageNet weights, fine-tuned (encoder lr 1e-5) |
| `clip` | CLIP ViT-B/16, 224 px, shared by both cameras | OpenAI CLIP weights, fine-tuned (encoder lr 1e-5) |

```bash
wandb login                                             # once per machine; or set wandb.mode=offline
make smoke-dp key_lock                                  # a few steps of every encoder: data, training, eval, checkpoint, W&B
make train-dp key_lock ENCODER=imagenet                 # full training (config.yaml defaults)
make train-dp key_lock ENCODER=clip ARGS="encoders.clip.batch_size=32 training.epochs=100"
make eval-dp CKPT=outputs/baselines/diffusion_policy/<run>/best.ckpt EPISODES=50
```

- The first run builds a lossless zarr replay buffer at `demos/<EnvId>/baseline_zarr_<hz>hz/` from the raw recording.
- Every `evaluation.every` epochs the EMA policy runs `evaluation.episodes` rollouts on reset seeds from
  `evaluation.seed_offset` (never used for demos), logging success rate, each failure flag's rate and videos.
- Runs use CUDA with bfloat16 when available and fall back to CPU. Pretrained weights download on first use
  (torchvision and Hugging Face caches).
- On an 8 GB GPU: `scratch`/`imagenet` use about 5 GB and take about 1 minute per epoch on 100 demos;
  `clip` uses about 5 GB at batch 16 and about 14 minutes per epoch, so train it on a larger GPU with a larger batch.

## pi0.5 (`pi05/`)

Fine-tunes Physical Intelligence's pi0.5 base checkpoint with [openpi](https://github.com/Physical-Intelligence/openpi),
installed by `make build-pi05` at a pinned commit in its own uv environment (`pi05/openpi/`, not committed), because
openpi pins JAX, PyTorch and NumPy versions that differ from this repository's.
The policy sees the base and wrist cameras, 7 arm joints plus the finger opening, and the task prompt; it predicts
20 actions (1 s at the 20 Hz export) of joint deltas plus an absolute gripper command, returned as absolute
joint targets, and the evaluator executes 10 before replanning with `pd_joint_pos`.

| `PI05_CONFIG` | Model | GPU memory (openpi's estimate) |
|---|---|---|
| `pi05_fr3_lora` (default) | pi0.5 base, LoRA on both Gemma models | about 22.5 GB |
| `pi05_fr3` | pi0.5 base, full fine-tune | about 70 GB |
| `pi05_fr3_smoke` | randomly initialized tiny Gemma with the full SigLIP encoder, 10 steps | runs on CPU |

```bash
make build-pi05                                   # clone openpi, uv sync its lockfile, check JAX sees the GPU
make smoke-pi05 key_lock                          # 3 episodes: convert, norm stats, 10 CPU steps, W&B, checkpoint, serve, rollout
make convert-pi05 key_lock                        # demos/<EnvId>/export_hdf5_20hz -> demos/lerobot/fr3/key_lock
make train-pi05 key_lock EXP=lora_s42 TRAIN_SEED=42   # norm stats on first run; repeat for 43 and 44
make serve-pi05 key_lock CKPT=outputs/baselines/pi05/checkpoints/pi05_fr3_lora/lora_v1/29999
make eval-pi05 key_lock EPISODES=50               # in a second terminal, while the server runs
```

- Recording must use `export.format: hdf5` with `joint_position` in `export.actions` (`config/data.yaml`).
- The pi0.5 base weights (about 12 GB) download from openpi's public bucket on the first training run.
- Checkpoints (with their norm stats) go to `outputs/baselines/pi05/checkpoints/<config>/<EXP>/<step>/`;
  `ARGS="--overwrite"` replaces an existing `EXP`, and evaluation results go to `outputs/baselines/pi05/eval/`.
- The policy can be served from a separate GPU machine: `make eval-pi05 key_lock HOST=<server ip> PORT=8000`.
