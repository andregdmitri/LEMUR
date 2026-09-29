#!/bin/bash

echo "Starting retfound..."

python main.py --run retfound_finetune --seed 42 --mask_ratio 0.0 --dataset mbrset
python main.py --run retfound_finetune --seed 43 --mask_ratio 0.0 --dataset mbrset
python main.py --run retfound_finetune --seed 44 --mask_ratio 0.0 --dataset mbrset
python main.py --run retfound_finetune --seed 45 --mask_ratio 0.0 --dataset mbrset
python main.py --run retfound_finetune --seed 46 --mask_ratio 0.0 --dataset mbrset
python main.py --run retfound_finetune --seed 42 --mask_ratio 0.0 --dataset messidor
python main.py --run retfound_finetune --seed 43 --mask_ratio 0.0 --dataset messidor
python main.py --run retfound_finetune --seed 44 --mask_ratio 0.0 --dataset messidor
python main.py --run retfound_finetune --seed 45 --mask_ratio 0.0 --dataset messidor
python main.py --run retfound_finetune --seed 46 --mask_ratio 0.0 --dataset messidor

echo "All models finished training!"