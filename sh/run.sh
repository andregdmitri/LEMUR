#!/bin/bash

echo "Starting unet..."
python main.py --run unet --seed 42 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 43 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 44 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 45 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 46 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 42 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 43 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 44 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 45 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 46 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize

echo "Starting efficientnet..."
python main.py --run efficientnet --seed 42 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 43 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 44 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 45 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 46 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 42 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 43 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 44 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 45 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 46 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize


echo "Starting mobilenet..."
python main.py --run mobilenet --seed 42 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 43 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 44 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 45 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 46 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 42 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 43 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 44 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 45 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 46 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize


echo "Starting vmamba..."
python main.py --run vmamba --seed 42 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 43 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 44 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 45 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 46 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 42 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 43 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 44 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 45 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 46 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize


# echo "Starting head distillation 1..."
# python main.py --run head --seed 42 --mask_ratio 0.0 --dataset idrid --load_backbone /exp/andremitri/checkpoints/g26bmtoi/checkpoints/best_distillation_42_aptos.ckpt --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run head --seed 43 --mask_ratio 0.0 --dataset idrid --load_backbone /exp/andremitri/checkpoints/g26bmtoi/checkpoints/best_distillation_43_aptos.ckpt --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run head --seed 44 --mask_ratio 0.0 --dataset idrid --load_backbone /exp/andremitri/checkpoints/g26bmtoi/checkpoints/best_distillation_44_aptos.ckpt --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run head --seed 45 --mask_ratio 0.0 --dataset idrid --load_backbone /exp/andremitri/checkpoints/g26bmtoi/checkpoints/best_distillation_45_aptos.ckpt --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run head --seed 46 --mask_ratio 0.0 --dataset idrid --load_backbone /exp/andremitri/checkpoints/g26bmtoi/checkpoints/best_distillation_46_aptos.ckpt --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run head --seed 42 --mask_ratio 0.0 --dataset aptos --load_backbone /exp/andremitri/checkpoints/g26bmtoi/checkpoints/best_distillation_42_aptos.ckpt --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run head --seed 43 --mask_ratio 0.0 --dataset aptos --load_backbone /exp/andremitri/checkpoints/g26bmtoi/checkpoints/best_distillation_43_aptos.ckpt --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run head --seed 44 --mask_ratio 0.0 --dataset aptos --load_backbone /exp/andremitri/checkpoints/g26bmtoi/checkpoints/best_distillation_44_aptos.ckpt --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run head --seed 45 --mask_ratio 0.0 --dataset aptos --load_backbone /exp/andremitri/checkpoints/g26bmtoi/checkpoints/best_distillation_45_aptos.ckpt --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run head --seed 46 --mask_ratio 0.0 --dataset aptos --load_backbone /exp/andremitri/checkpoints/g26bmtoi/checkpoints/best_distillation_46_aptos.ckpt --augmentation retina_all --num_workers 8 --prune --quantize


# echo "Starting retfound_finetune..."
# python main.py --run retfound_finetune --seed 46 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize

echo "All models finished training!"