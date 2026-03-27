# constants.py
import os
import torch
from dotenv import load_dotenv

load_dotenv()

# Model Config
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 64))
DIST_EPOCHS = int(os.getenv('DIST_EPOCHS', 1000))
HEAD_EPOCHS = int(os.getenv('HEAD_EPOCHS', 1000))
MASK_RATIO = float(os.getenv('MASK_RATIO', 0.75))

INPUT_DIM = int(os.getenv('INPUT_DIM', 784))
NUM_CLASSES = int(os.getenv('NUM_CLASSES', 5))
IMG_SIZE = int(os.getenv('IMG_SIZE', 224))
IN_CHANS = int(os.getenv('IN_CHANS', 3))
VMAMBA_DEPTH = int(os.getenv('VMAMBA_DEPTH', 4)) # num of blocks  ### 4, 8, 16,
VMAMBA_EMBED_DIM = int(os.getenv('VMAMBA_EMBED_DIM', 32)) # projector -> teacher ### 32, 64, 128,
SSM_DIM = int(os.getenv('SSM_DIM', 4)) # spatial mixing module dimension ### 4, 8, 16,
EXPAND_DIM = int(os.getenv('EXPAND_DIM', VMAMBA_EMBED_DIM * 2)) ### 64, 128, 256,
PATCH_SIZE = int(os.getenv('PATCH_SIZE', 32)) 
TEACHER_EMBED_DIM = int(os.getenv('TEACHER_EMBED_DIM', 1024))
NUM_WORKERS = int(os.getenv('NUM_WORKERS', 12))
PATIENCE = int(os.getenv('PATIENCE', 50))
FREEZE_BACKBONE = False  # os.getenv('FREEZE_BACKBONE', 'False') == 'True'
STUDENT_MODEL = os.getenv('STUDENT_MODEL', 'vmamba')  # options: vmamba, tinyvit, mobilenet_v3_small, efficientnet_b0
SEED = int(os.getenv('SEED', 42))

# Optimizer settings
WARMUP_EPOCHS = int(os.getenv('WARMUP_EPOCHS', 10))
LR = float(os.getenv('LR', 5e-4))
FINAL_LR = float(os.getenv('FINAL_LR', 1e-6))
WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', 1e-4))

# Augmentation control
USE_MIXUP = os.getenv('USE_MIXUP', 'True') == 'True'
MIXUP_ALPHA = float(os.getenv('MIXUP_ALPHA', 0.4))
USE_MOSAIC = os.getenv('USE_MOSAIC', 'True') == 'True'
MOSAIC_PROB = float(os.getenv('MOSAIC_PROB', 0.3))
USE_COPY_PASTE = os.getenv('USE_COPY_PASTE', 'True') == 'True'
COPY_PASTE_PROB = float(os.getenv('COPY_PASTE_PROB', 0.3))

# Other settings
DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

# Paths
DATA_DIR = os.getenv('DATA_DIR', './data')
CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR', './checkpoints')
IDRID_PATH = os.getenv('IDRID_PATH', os.path.join(DATA_DIR, 'aaryapatel98/indian-diabetic-retinopathy-image-dataset/versions/1/B.%20Disease%20Grading/B. Disease Grading'))
APTOS_PATH = os.getenv('APTOS_PATH', os.path.join(DATA_DIR, 'aptos2019/versions/3'))
MBRSET_PATH = os.getenv('MBRSET_PATH', os.path.join(DATA_DIR, 'mbrset/1'))
