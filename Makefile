# Usage: make <target> [task], e.g. `make demos key_lock` or `make smoke button_cover EPISODES=5`.

TASKS := ball_valve ball_valve_ring circuit_breaker gate_valve_small gate_valve_large button_cover key_lock drawer peg_insertion thread_m8 thread_m30 light_bulb shock_absorber
TASK ?= $(firstword $(filter $(TASKS),$(MAKECMDGOALS)))
UV := $(or $(shell command -v uv 2>/dev/null),$(HOME)/.local/bin/uv)
PY := $(UV) run --locked python

N ?=
SEED ?= 0
# Training seed for train-dp and train-pi05; docs/RESEARCH.md asks for 3 per method.
TRAIN_SEED ?= 42
EPISODES ?= 3
RECORD_DIR ?= demos
ANNOTATE_DIR ?=
VIDEO_DIR ?= outputs/videos
DISTRACTORS ?= 0
# Diffusion Policy baseline: scratch | imagenet | clip; ARGS passes extra config overrides, e.g. ARGS="training.epochs=100".
ENCODER ?= scratch
ENCODERS := scratch imagenet clip
ARGS ?=
# pi0.5 baseline (openpi in its own uv environment): pi05_fr3_lora | pi05_fr3 | pi05_fr3_smoke; EXP names the checkpoint folder.
OPENPI_DIR := baselines/pi05/openpi
OPENPI_COMMIT := 215abfb217dbac7d5f1273282331b9b1866c0479
PI05_CONFIG ?= pi05_fr3_lora
HZ ?= 20
EXP ?= $(TASK)_s$(TRAIN_SEED)
PORT ?= 8000
HOST ?= localhost
PI05 := HF_LEROBOT_HOME=$(CURDIR)/demos/lerobot $(UV) run --project $(OPENPI_DIR) --locked python

.DEFAULT_GOAL := help
.PHONY: help build check run smoke smoke-all demos view video annotate eval train-dp smoke-dp eval-dp build-pi05 convert-pi05 train-pi05 serve-pi05 eval-pi05 smoke-pi05 clean $(TASKS)

help:
	@echo "make build              install uv, Python 3.12 and the locked dependencies, then verify rendering"
	@echo "make check              verify task registration and headless camera rendering"
	@echo "make run                run the FR3 table viewer (scripts/run.py)"
	@echo "make smoke <task>       smoke test (EPISODES=$(EPISODES), DISTRACTORS=$(DISTRACTORS))"
	@echo "make smoke-all          smoke test every task"
	@echo "make demos <task>       record successful demos to $(RECORD_DIR)/ and export them (count, format, parallel robots: config/data.yaml; N= overrides)"
	@echo "make view <task>        watch scripted demos live in the SAPIEN viewer (N=1)"
	@echo "make video <task>       record one successful demo video to $(VIDEO_DIR)/"
	@echo "make annotate           select a rollout folder and mark failure onset/end (RECORD_DIR=... optional)"
	@echo "make eval                evaluate the runs in config/eval.yaml into outputs/eval/results_<time>.json (ARGS=\"--runs name\")"
	@echo "make train-dp <task>    train Diffusion Policy (ENCODER=$(ENCODER): scratch | imagenet | clip, ARGS=overrides)"
	@echo "make smoke-dp <task>    tiny Diffusion Policy run for every encoder (data, training, eval, checkpoints, W&B)"
	@echo "make eval-dp CKPT=...   evaluate a Diffusion Policy checkpoint (EPISODES=50)"
	@echo "make build-pi05         install openpi (pinned commit) in its own uv environment"
	@echo "make convert-pi05 <task> convert the task's hdf5 export to a LeRobot dataset (fr3/<task>)"
	@echo "make train-pi05 <task>  fine-tune pi0.5 base (PI05_CONFIG=$(PI05_CONFIG), EXP=<name>, ARGS=\"--steps 1000\")"
	@echo "make serve-pi05 <task> CKPT=<step dir>   serve a checkpoint on PORT=$(PORT)"
	@echo "make eval-pi05 <task>   evaluate the served policy in ManiSkill (EPISODES=50)"
	@echo "make smoke-pi05 <task>  tiny pi0.5 on CPU: convert, train, checkpoint, W&B, serve and a short rollout"
	@echo "make clean              remove outputs/ and Python caches"
	@echo "tasks: $(TASKS)"

build:
	@command -v uv >/dev/null 2>&1 || test -x "$(HOME)/.local/bin/uv" || curl -LsSf https://astral.sh/uv/install.sh | sh
	$(UV) python install
	$(UV) sync --locked
	@test -n "$$(ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null)" || { \
		echo "No Vulkan driver found; SAPIEN rendering needs one."; \
		echo "Ubuntu: sudo apt install libvulkan1 mesa-vulkan-drivers (plus the NVIDIA driver on NVIDIA GPUs)"; \
		exit 1; }
	$(MAKE) --no-print-directory check

check:
	$(PY) -c "import tasks; print('registered:', ', '.join(tasks.TASKS.values()))"
	$(PY) -m scripts.run --headless --steps 5

run:
	$(PY) -m scripts.run

smoke: require-task
	$(PY) -m tasks.smoke_test --task $(TASK) --episodes $(EPISODES) $(if $(filter 1,$(DISTRACTORS)),--distractors)

smoke-all:
	@for task in $(TASKS); do $(MAKE) --no-print-directory smoke TASK=$$task || exit 1; done

demos: require-task
	$(PY) -m tasks.generate_demos --task $(TASK) $(if $(N),-n $(N)) --seed $(SEED) --record-dir $(RECORD_DIR)

view: require-task
	$(PY) -m tasks.generate_demos --task $(TASK) -n $(or $(N),1) --seed $(SEED) --record-dir $(VIDEO_DIR)/viewer --viewer

video: require-task
	$(PY) -m tasks.generate_demos --task $(TASK) -n 1 --max-attempts 10 --record-dir $(VIDEO_DIR) --save-video

annotate:
	$(PY) -m scripts.annotate_rollouts $(if $(ANNOTATE_DIR),--record-dir "$(ANNOTATE_DIR)",$(if $(filter command line,$(origin RECORD_DIR)),--record-dir "$(RECORD_DIR)",))

train-dp: require-task
	$(PY) -m baselines.diffusion_policy.train task=$(TASK) encoder=$(ENCODER) training.seed=$(TRAIN_SEED) $(ARGS)

smoke-dp: require-task
	@for encoder in $(ENCODERS); do $(PY) -m baselines.diffusion_policy.train --smoke task=$(TASK) encoder=$$encoder $(ARGS) || exit 1; done

eval:
	$(PY) -m scripts.evaluate $(ARGS)

eval-dp:
	@test -n "$(CKPT)" || { echo "set CKPT=path/to/best.ckpt"; exit 1; }
	$(PY) -m baselines.diffusion_policy.evaluate --checkpoint $(CKPT) --episodes $(if $(filter command line,$(origin EPISODES)),$(EPISODES),50)

build-pi05:
	@test -d $(OPENPI_DIR)/.git || GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Physical-Intelligence/openpi.git $(OPENPI_DIR)
	cd $(OPENPI_DIR) && git fetch -q origin && git checkout -q $(OPENPI_COMMIT) && git submodule update --init --recursive
	cd $(OPENPI_DIR) && GIT_LFS_SKIP_SMUDGE=1 $(UV) sync --locked
	$(PI05) -c "import jax, baselines.pi05.config; print('openpi ready, jax devices:', jax.devices())"

convert-pi05: require-task
	@env_id=$$($(PY) -c "from tasks import TASKS; print(TASKS['$(TASK)'])" 2>/dev/null | tail -1); \
	$(PI05) -m baselines.pi05.convert $(RECORD_DIR)/$$env_id/export_hdf5_$(HZ)hz --repo-id fr3/$(TASK) $(ARGS)

train-pi05: require-task
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 $(PI05) -m baselines.pi05.train $(PI05_CONFIG) --repo-id fr3/$(TASK) --exp-name $(EXP) --seed $(TRAIN_SEED) $(ARGS)

serve-pi05: require-task
	@test -n "$(CKPT)" || { echo "set CKPT=outputs/baselines/pi05/checkpoints/$(PI05_CONFIG)/$(EXP)/<step>"; exit 1; }
	$(PI05) -m baselines.pi05.serve $(PI05_CONFIG) --checkpoint $(CKPT) --repo-id fr3/$(TASK) --port $(PORT)

eval-pi05: require-task
	$(PY) -m baselines.pi05.evaluate --task $(TASK) --host $(HOST) --port $(PORT) --hz $(HZ) --episodes $(if $(filter command line,$(origin EPISODES)),$(EPISODES),50)

# The smoke model still carries pi0.5's full SigLIP encoder, so it runs JAX on CPU to fit any machine.
smoke-pi05: require-task
	@env_id=$$($(PY) -c "from tasks import TASKS; print(TASKS['$(TASK)'])" 2>/dev/null | tail -1); \
	$(PI05) -m baselines.pi05.convert $(RECORD_DIR)/$$env_id/export_hdf5_$(HZ)hz --repo-id fr3/$(TASK)_smoke --max-episodes 3 && \
	JAX_PLATFORMS=cpu $(PI05) -m baselines.pi05.train pi05_fr3_smoke --repo-id fr3/$(TASK)_smoke --exp-name $(TASK) --overwrite && \
	{ JAX_PLATFORMS=cpu $(PI05) -m baselines.pi05.serve pi05_fr3_smoke --repo-id fr3/$(TASK)_smoke --port 8765 \
		--checkpoint outputs/baselines/pi05/checkpoints/pi05_fr3_smoke/$(TASK)/9 & server=$$!; \
	trap 'kill $$server 2>/dev/null' EXIT; \
	$(PY) -m baselines.pi05.evaluate --task $(TASK) --port 8765 --hz $(HZ) --episodes 1 --max-steps 30 --videos 1; }

clean:
	rm -rf outputs
	find . -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +

.PHONY: require-task
require-task:
	@test -n "$(TASK)" || { echo "missing task; one of: $(TASKS)"; exit 1; }

# Task names given as extra goals only select the task.
$(TASKS):
	@:
