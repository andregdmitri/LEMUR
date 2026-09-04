#!/bin/bash

echo "Starting unet..."
python main.py --run unet --seed 42 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 43 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 44 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 45 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 46 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 42 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 43 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 44 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 45 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 46 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 42 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 43 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 44 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 45 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run unet --seed 46 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize


echo "Starting efficientnet..."
python main.py --run efficientnet --seed 42 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 43 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 44 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 45 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 46 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 42 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 43 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 44 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 45 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 46 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 42 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 43 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 44 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 45 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run efficientnet --seed 46 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize


echo "Starting mobilenet..."
python main.py --run mobilenet --seed 42 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 43 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 44 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 45 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 46 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 42 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 43 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 44 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 45 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 46 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 42 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 43 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 44 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 45 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run mobilenet --seed 46 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize


echo "Starting vmamba..."
python main.py --run vmamba --seed 42 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 43 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 44 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 45 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 46 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 42 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 43 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 44 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 45 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 46 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 42 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 43 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 44 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 45 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run vmamba --seed 46 --mask_ratio 0.0 --dataset mured --augmentation retina_all --num_workers 8 --prune --quantize


# echo "Starting retfound_finetune..."
# python main.py --run retfound_finetune --seed 46 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize

echo "All models finished training!"