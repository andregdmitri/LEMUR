from torchvision import transforms
import cv2
import numpy as np
import torch
import random
from PIL import Image
from config.constants import NUM_CLASSES


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
    # images: list of 4 tensors [C,H,W]
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

    # blend labels proportional ao patch area
    alpha = (patch_h * patch_w) / (h * w)
    y = (1 - alpha) * one_hot(y1) + alpha * one_hot(y2)
    return x, y


def preprocess_image(img_path):
    """
    Applies the preprocessing pipeline from the paper:
    Cropping -> Green Channel -> Median Filter -> CLAHE -> Bilateral Filter
    """
    # Load image with OpenCV
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Crop partial black background
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # Add a small padding if desired, otherwise strictly use bounding box
        img = img[y:y+h, x:x+w]

    # 2. Extract Green Channel (RGB -> Index 1)
    green_channel = img[:, :, 1]

    # 3. Median filter (kernel size 3 or 5 is standard, using 3 here)
    median = cv2.medianBlur(green_channel, 3)

    # 4. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(median)

    # 5. Bilateral filter (d, sigmaColor, sigmaSpace)
    bilateral = cv2.bilateralFilter(clahe_img, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert back to PIL Image. 
    # We duplicate the grayscale channel to 3 channels (RGB) so it doesn't break 
    # standard PyTorch transforms and pre-trained models expecting 3 input channels.
    processed_img_rgb = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(processed_img_rgb)


def eval_transform(img_size):
    """Transforms used for evaluation / validation: deterministic resize + to-tensor."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def train_transform_default(img_size):
    """Basic train transform (no strong augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def train_transform_retfound_linear(img_size):
    """Stronger augmentations used for RETFound linear probing mode."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
    ])


def train_transform_strong(img_size):
    """Strong augmentations for training (rotation, flips, affine, crop)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=90), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Handles translations
        transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])