# LEMUR: A Ultra-Lightweight Distilled Retinal Foundation Model Using Mamba

This repository contains a unified pipeline for distilling a student VMamba model from a RETFound teacher, training classification heads, and evaluating on retinal fundus datasets (IDRiD, APTOS, MBRSET).

This README focuses on the CLI (`main.py`) and the actual project layout so you can run evaluation and training quickly.

---

## Repository layout (actual)

- `main.py` — unified CLI entrypoint (see CLI section below)
- `config/` — `constants.py` (defaults and environment-driven overrides)
- `dataloader/` — `idrid.py`, `aptos.py`, `mbrset.py` (Lightning DataModules with preprocessing)
- `models/` — `retfound.py`, `vmamba_backbone.py`, `dist.py`, `models_vit.py`
- `train/` — `distill.py`, `head.py`, `train_retfound.py`
- `eval/` — `eval_vmamba.py`, `eval_retfound.py`, `shared_eval.py`
- `optimizers/` — `optimizer.py` (helper for warmup + cosine schedule)
- `utils/` — utilities (`flops.py`, `pos_embed.py`, `transforms.py` with preprocessing)
- `imgs/`, `results/` — example outputs and CSVs
- `requirements.txt`, `simple_test.ipynb`

---

## Data Preprocessing

All datasets (IDRiD, APTOS, MBRSET) apply the following preprocessing pipeline to each image:

1. **Crop partial black background**: Automatic contour detection to remove black borders.
2. **Green Channel Extraction**: Retain only the green channel (common in retinal imaging).
3. **Median Filter**: 3x3 kernel to reduce noise.
4. **CLAHE**: Contrast Limited Adaptive Histogram Equalization (clipLimit=2.0, tileGridSize=8x8).
5. **Bilateral Filter**: Edge-preserving smoothing (d=9, sigmaColor=75, sigmaSpace=75).
6. **Convert to RGB**: Duplicate grayscale to 3-channel RGB for compatibility with standard models.

This preprocessing is implemented in `utils/transforms.py` and applied in all dataloaders.

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

### Advanced models and comparison setup

This repo now supports distillation into multiple student architectures for direct performance and efficiency comparison in fundus imaging:

- `vmamba` (default): lightweight Vision Mamba distill student (MAE-style mask-based training)
- `tinyvit`: small transformer student from `timm`
- `mobilenet_v3_small`: CNN student from `timm`
- `efficientnet_b0`: CNN student from `timm`

Use `--student_model` in CLI to switch student architecture:

```bash
python main.py --run distill --student_model tinyvit --mask_ratio 0.0
python main.py --run head --student_model mobilenet_v3_small --load_backbone CHECKPOINT
python main.py --run eval --student_model efficientnet_b0 --load_model CHECKPOINT
```

### Masking clarification
- During distillation, the teacher is always fed the full image (unmasked).
- Masking is applied only to the student input when the student supports it (VMamba in this repository). For non-MAEs (TinyViT/MobileNet/EfficientNet) masking is disabled.

### Lightweight / efficient model literature (context)
- VMamba distilled from RETFound is inspired by modern retinal foundation model distillation (SOTA in fundoscopy for low-resource deployment).
- Additional student baselines include MobileNetV3 and EfficientNet-B0, plus a lightweight transformer student (TinyViT), for direct resource-vs-accuracy tradeoff comparisons.
- References (2, 3, 4): please include your key papers on transformer distillation, deployable ophthalmic AI, and lightweight CNN backbones in your manuscript.

### Pruning & Quantization
- A new helper module is available at `train/prune_quantize.py`.
- Use structured pruning and dynamic quantization for efficient deployment targets (CPU, edge devices).

# Checkpoints & formats

- VMamba head output (`train/head.py`) saves a file expected by `eval` that contains two keys: `backbone` and `head` (each a state_dict).
- RETFound evaluation expects a PyTorch Lightning checkpoint compatible with `train/train_retfound.RETFoundTask`.
