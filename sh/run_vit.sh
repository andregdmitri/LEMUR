#!/bin/bash

# echo "Starting tinyvit..."
# python main.py --run tinyvit --seed 42 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run tinyvit --seed 43 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run tinyvit --seed 44 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run tinyvit --seed 45 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run tinyvit --seed 46 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize

# python main.py --run tinyvit --seed 42 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run tinyvit --seed 43 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run tinyvit --seed 44 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run tinyvit --seed 45 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run tinyvit --seed 46 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize

# python main.py --run tinyvit --seed 42 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run tinyvit --seed 43 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run tinyvit --seed 44 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run tinyvit --seed 45 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run tinyvit --seed 46 --mask_ratio 0.0 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize

# python main.py --run tinyvit --seed 42 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run tinyvit --seed 43 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run tinyvit --seed 44 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run tinyvit --seed 45 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run tinyvit --seed 46 --mask_ratio 0.0 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize


# echo "Starting distill tinyvit..."
# python main.py --run distill --student tinyvit --seed 42 --mask_ratio 0.75 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run distill --student tinyvit --seed 43 --mask_ratio 0.75 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run distill --student tinyvit --seed 44 --mask_ratio 0.75 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run distill --student tinyvit --seed 45 --mask_ratio 0.75 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run distill --student tinyvit --seed 46 --mask_ratio 0.75 --dataset idrid --augmentation retina_all --num_workers 8 --prune --quantize

# python main.py --run distill --student tinyvit --seed 42 --mask_ratio 0.75 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run distill --student tinyvit --seed 43 --mask_ratio 0.75 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run distill --student tinyvit --seed 44 --mask_ratio 0.75 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run distill --student tinyvit --seed 45 --mask_ratio 0.75 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize
# python main.py --run distill --student tinyvit --seed 46 --mask_ratio 0.75 --dataset aptos --augmentation retina_all --num_workers 8 --prune --quantize

python main.py --run distill --student tinyvit --seed 42 --mask_ratio 0.75 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run distill --student tinyvit --seed 43 --mask_ratio 0.75 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run distill --student tinyvit --seed 44 --mask_ratio 0.75 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run distill --student tinyvit --seed 45 --mask_ratio 0.75 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run distill --student tinyvit --seed 46 --mask_ratio 0.75 --dataset mbrset --augmentation retina_all --num_workers 8 --prune --quantize

python main.py --run distill --student tinyvit --seed 42 --mask_ratio 0.75 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run distill --student tinyvit --seed 43 --mask_ratio 0.75 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run distill --student tinyvit --seed 44 --mask_ratio 0.75 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run distill --student tinyvit --seed 45 --mask_ratio 0.75 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize
python main.py --run distill --student tinyvit --seed 46 --mask_ratio 0.75 --dataset messidor --augmentation retina_all --num_workers 8 --prune --quantize


echo "All models finished training!"