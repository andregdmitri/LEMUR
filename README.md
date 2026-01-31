# LEMUR: A Ultra-Lightweight Distilled Retinal Foundation Model Using Mamba

This repository contains a unified pipeline for distilling a student VMamba model from a RETFound teacher, training classification heads, and evaluating on retinal fundus datasets (IDRiD, APTOS, MBRSET).

This README focuses on the CLI (`main.py`) and the actual project layout so you can run evaluation and training quickly.

---

## Repository layout (actual)

- `main.py` — unified CLI entrypoint (see CLI section below)
- `config/` — `constants.py` (defaults and environment-driven overrides)
- `dataloader/` — `idrid.py`, `aptos.py` (Lightning DataModules)
- `models/` — `retfound.py`, `vmamba_backbone.py`, `dist.py`, `models_vit.py`
- `train/` — `distill.py`, `head.py`, `train_retfound.py`
- `eval/` — `eval_vmamba.py`, `eval_retfound.py`, `shared_eval.py`
- `optimizers/` — `optimizer.py` (helper for warmup + cosine schedule)
- `utils/` — utilities (`flops.py`, `pos_embed.py`, etc.)
- `imgs/`, `results/` — example outputs and CSVs
- `requirements.txt`, `simple_test.ipynb`

---

## CLI (`main.py`) — modes and arguments

Run `python main.py --run <mode> [options]`.

Modes (value for `--run`):

- `distill` — Phase I: distill RETFound teacher -> VMamba student
- `head` — Phase II: train classification head on distilled backbone
- `eval` — Evaluate VMamba (expects a checkpoint saved by head training)
- `retfound_linear` — RETFound linear probe (uses `train_retfound.py`)
- `retfound_finetune` — RETFound fine-tuning
- `retfound_eval` — Evaluate a RETFound Lightning checkpoint

Shared CLI arguments (most common):

- `--lr` — learning rate (default from `config/constants.py`)
- `--mask_ratio` — mask ratio used during distillation
- `--dist_epochs` — number of distillation epochs
- `--head_epochs` — number of head training epochs
- `--teacher_ckpt` — optional teacher checkpoint override
- `--load_backbone` — path to distilled backbone (required for `--run head`)
- `--load_model` — path to full model checkpoint for evaluation (required for `--run eval` and `--run retfound_eval`)
- `--dataset` — `idrid` (default) or `aptos`

RETFound-specific:

- `--checkpoint` — RETFound pretrained checkpoint (optional)
- `--epochs` — epochs for RETFound linear/finetune modes

The CLI enforces required flags per mode (e.g., `--load_backbone` for `head`). See `main.py` for the full help text.

---

## Common commands / examples

Evaluation — VMamba (IDRiD):

```bash
python main.py --run eval --load_model CHECKPOINT_PATH --dataset idrid
```

Evaluation — VMamba (MBRSET):

```bash
python main.py --run eval --load_model CHECKPOINT_PATH --dataset mbrset
```

Evaluation — VMamba (APTOS):

```bash
python main.py --run eval --load_model CHECKPOINT_PATH --dataset aptos
```

Evaluation — RETFound Lightning ckpt:

```bash
python main.py --run retfound_eval --load_model checkpoints/retfound_finetuned.ckpt --dataset idrid
```

Distillation (Phase I):

```bash
python main.py --run distill --lr 1e-4 --mask_ratio 0.75
```

Head training (Phase II):

```bash
python main.py --run head --load_backbone checkpoints/vmamba_distilled_student.pth --lr 1e-4
```

RETFound linear / finetune:

```bash
# Linear probe
python main.py --run retfound_linear --dataset idrid --lr 3e-4

# Fine-tune
python main.py --run retfound_finetune --dataset aptos --lr 1e-5
```

---

## Checkpoints & formats

- VMamba head output (`train/head.py`) saves a file expected by `eval` that contains two keys: `backbone` and `head` (each a state_dict).
- RETFound evaluation expects a PyTorch Lightning checkpoint compatible with `train/train_retfound.RETFoundTask`.
