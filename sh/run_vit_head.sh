#!/bin/bash

echo "Starting student tinyvit..."
python main.py --run head --student tinyvit --seed 42 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/y9toj4cb/checkpoints/best_distillation_42_idrid.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student tinyvit --seed 43 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/6ie1snyl/checkpoints/best_distillation_43_idrid.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student tinyvit --seed 44 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/4g453cep/checkpoints/best_distillation_44_idrid.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student tinyvit --seed 45 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/qm450dag/checkpoints/best_distillation_45_idrid.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student tinyvit --seed 46 --mask_ratio 0.0 --dataset idrid --augmentation retina_all --load_backbone ./distillation/wkh9n1d0/checkpoints/best_distillation_46_idrid.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student tinyvit --seed 42 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/o14oto4d/checkpoints/best_distillation_42_aptos.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student tinyvit --seed 43 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/ol63ajcp/checkpoints/best_distillation_43_aptos.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student tinyvit --seed 44 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/za5sz9br/checkpoints/best_distillation_44_aptos.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student tinyvit --seed 45 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/4lph1t0d/checkpoints/best_distillation_45_aptos.ckpt --num_workers 8 --prune --quantize
python main.py --run head --student tinyvit --seed 46 --mask_ratio 0.0 --dataset aptos --augmentation retina_all --load_backbone ./distillation/zbws26n1/checkpoints/best_distillation_46_aptos.ckpt --num_workers 8 --prune --quantize


echo "All models finished training!"