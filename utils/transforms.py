from torchvision import transforms
import cv2
import numpy as np
import torch
import random
from PIL import Image
from config.constants import NUM_CLASSES

# --- Standard ImageNet Normalization ---
# If using pre-trained models, these values are standard. 
# For optimal performance, consider calculating mean/std of your specific dataset.
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def one_hot(label, num_classes=NUM_CLASSES):
    if isinstance(label, torch.Tensor):
        label = int(label.item())
    vec = torch.zeros(num_classes, dtype=torch.float32)
    vec[label] = 1.0
    return vec

def mixup_data(x1, x2, y1, y2, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    x = lam * x1 + (1.0 - lam) * x2
    y = lam * one_hot(y1) + (1.0 - lam) * one_hot(y2)
    return x, y, lam

def mosaic_data(images, labels):
    h, w = images[0].shape[1:]
    h2, w2 = h // 2, w // 2
    out = torch.zeros_like(images[0])
    out[:, :h2, :w2] = images[0][:, :h2, :w2]
    out[:, :h2, w2:] = images[1][:, :h2, :w2]
    out[:, h2:, :w2] = images[2][:, :h2, :w2]
    out[:, h2:, w2:] = images[3][:, :h2, :w2]
    y = (one_hot(labels[0]) + one_hot(labels[1]) + one_hot(labels[2]) + one_hot(labels[3])) / 4.0
    return out, y

def copy_paste_data(x1, x2, y1, y2):
    _, h, w = x1.shape
    patch_h = h // 3
    patch_w = w // 3
    xs = random.randint(0, h - patch_h)
    ys = random.randint(0, w - patch_w)

    xp = x2[:, xs:xs + patch_h, ys:ys + patch_w].clone()
    x = x1.clone()
    x[:, xs:xs + patch_h, ys:ys + patch_w] = xp

    alpha = (patch_h * patch_w) / (h * w)
    y = (1 - alpha) * one_hot(y1) + alpha * one_hot(y2)
    return x, y

def preprocess_image(img_path, sigmaX=30):
    """
    Improved preprocessing with Luminance Normalization (Ben Graham method)
    and artifact reduction.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Background Cropping (Your existing robust logic)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        img = img[y:y+h, x:x+w]

    # 2. Extract Green Channel
    green_channel = img[:, :, 1]

    # 3. Luminance Normalization
    # This subtracts the blurred version to remove uneven illumination
    blurred = cv2.GaussianBlur(green_channel, (0, 0), sigmaX)
    # The weight 4 and -4 are standard in top Kaggle solutions for DR
    normalized = cv2.addWeighted(green_channel, 4, blurred, -4, 128)

    # 4. CLAHE (Local Contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(normalized)

    # 5. Optional: Circular Masking 
    # If your dataset has black corners after cropping, this removes them
    # to prevent the model from learning "corner artifacts"
    h, w = clahe_img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(w, h) * 0.45), 255, -1)
    clahe_img = cv2.bitwise_and(clahe_img, clahe_img, mask=mask)

    # Convert back to 3-channel RGB for PyTorch models
    processed_img_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(processed_img_rgb)

def eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        NORMALIZE
    ])

def train_transform_default(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        NORMALIZE
    ])

def train_transform_retina_all(img_size):
    """
    State-of-the-art augmentation for retinal fundus.
    Combines RandAugment for policy optimization and ElasticTransform for anatomical robustness.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        # RandAugment is highly recommended for medical imaging datasets
        transforms.RandAugment(num_ops=2, magnitude=9),
        # Elastic deformation mimics variations in eye positioning/lens geometry
        transforms.RandomApply([transforms.ElasticTransform(alpha=50.0)], p=0.3),
        transforms.ToTensor(),
        # Coarse Dropout/Erasing
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        NORMALIZE
    ])

def train_transform_retina_strong(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=90),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.35),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0),
        transforms.ToTensor(),
        NORMALIZE
    ])

def build_train_transform(augmentation_mode, img_size):
    mode = (augmentation_mode or "default").lower()
    registry = {
        "default": train_transform_default,
        "none": train_transform_default,
        "retina_all": train_transform_retina_all,
        "retina_strong": train_transform_retina_strong,
    }
    if mode not in registry:
        raise ValueError(f"Unsupported mode '{augmentation_mode}'. Choose from {list(registry.keys())}.")
    return registry[mode](img_size)