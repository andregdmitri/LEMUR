#!/bin/bash

echo "Starting Dvmamba..."
python main.py --run head --seed 42 --mask_ratio 0.0 --dataset idrid --load_backbone /exp/andremitri/checkpoints/86ftw66j/checkpoints/best_distillation_42_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run head --seed 43 --mask_ratio 0.0 --dataset idrid --load_backbone /exp/andremitri/checkpoints/z0u2r1ry/checkpoints/best_distillation_43_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run head --seed 44 --mask_ratio 0.0 --dataset idrid --load_backbone /exp/andremitri/checkpoints/zfmcuc5c/checkpoints/best_distillation_44_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run head --seed 45 --mask_ratio 0.0 --dataset idrid --load_backbone /exp/andremitri/checkpoints/6yak0ad2/checkpoints/best_distillation_45_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run head --seed 46 --mask_ratio 0.0 --dataset idrid --load_backbone /exp/andremitri/checkpoints/qprbue1v/checkpoints/best_distillation_46_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize

python main.py --run head --seed 42 --mask_ratio 0.0 --dataset aptos --load_backbone /exp/andremitri/checkpoints/86ftw66j/checkpoints/best_distillation_42_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run head --seed 43 --mask_ratio 0.0 --dataset aptos --load_backbone /exp/andremitri/checkpoints/z0u2r1ry/checkpoints/best_distillation_43_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run head --seed 44 --mask_ratio 0.0 --dataset aptos --load_backbone /exp/andremitri/checkpoints/zfmcuc5c/checkpoints/best_distillation_44_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run head --seed 45 --mask_ratio 0.0 --dataset aptos --load_backbone /exp/andremitri/checkpoints/6yak0ad2/checkpoints/best_distillation_45_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run head --seed 46 --mask_ratio 0.0 --dataset aptos --load_backbone /exp/andremitri/checkpoints/qprbue1v/checkpoints/best_distillation_46_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize

python main.py --run head --seed 42 --mask_ratio 0.0 --dataset messidor --load_backbone /exp/andremitri/checkpoints/86ftw66j/checkpoints/best_distillation_42_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run head --seed 43 --mask_ratio 0.0 --dataset messidor --load_backbone /exp/andremitri/checkpoints/z0u2r1ry/checkpoints/best_distillation_43_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run head --seed 44 --mask_ratio 0.0 --dataset messidor --load_backbone /exp/andremitri/checkpoints/zfmcuc5c/checkpoints/best_distillation_44_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run head --seed 45 --mask_ratio 0.0 --dataset messidor --load_backbone /exp/andremitri/checkpoints/6yak0ad2/checkpoints/best_distillation_45_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run head --seed 46 --mask_ratio 0.0 --dataset messidor --load_backbone /exp/andremitri/checkpoints/qprbue1v/checkpoints/best_distillation_46_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize

python main.py --run head --seed 42 --mask_ratio 0.0 --dataset mbrset --load_backbone /exp/andremitri/checkpoints/86ftw66j/checkpoints/best_distillation_42_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run head --seed 43 --mask_ratio 0.0 --dataset mbrset --load_backbone /exp/andremitri/checkpoints/z0u2r1ry/checkpoints/best_distillation_43_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run head --seed 44 --mask_ratio 0.0 --dataset mbrset --load_backbone /exp/andremitri/checkpoints/zfmcuc5c/checkpoints/best_distillation_44_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run head --seed 45 --mask_ratio 0.0 --dataset mbrset --load_backbone /exp/andremitri/checkpoints/6yak0ad2/checkpoints/best_distillation_45_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run head --seed 46 --mask_ratio 0.0 --dataset mbrset --load_backbone /exp/andremitri/checkpoints/qprbue1v/checkpoints/best_distillation_46_aptos.ckpt  --augmentation retina_all --num_workers 8 --prune --quantize

echo "All models finished training!"