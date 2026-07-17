"""
Centralized dataloader utilities to avoid repetitive if/else logic across train files.
Provides unified functions for getting dataloaders and class weights.
"""

import os
import torch
from config.constants import *
from dataloader.idrid import IDRiDModule, compute_idrid_class_weights
from dataloader.aptos import APTOSModule
from dataloader.messidor import MessidorModule, compute_messidor_class_weights
from dataloader.mbrset import MBRSETModule, compute_mbrset_class_weights
from dataloader.papila import PAPILAModule, compute_papila_class_weights


DATASET_CONFIG = {
    "idrid": {
        "module": IDRiDModule,
        "path": IDRID_PATH,
        "weights_fn": compute_idrid_class_weights,
    },
    "aptos": {
        "module": APTOSModule,
        "path": APTOS_PATH,
        "weights_fn": None,
    },
    "messidor": {
        "module": MessidorModule,
        "path": MESSIDOR_PATH,
        "weights_fn": compute_messidor_class_weights,
    },
    "mbrset": {
        "module": MBRSETModule,
        "path": MBRSET_PATH,
        "weights_fn": compute_mbrset_class_weights,
    },
    "papila": {
        "module": PAPILAModule,
        "path": PAPILA_PATH,
        "weights_fn": compute_papila_class_weights,
    },
}


def get_dataloader(dataset_name, transform, batch_size=BATCH_SIZE):
    """
    Get a LightningDataModule for the specified dataset.
    
    Args:
        dataset_name: str, one of ['idrid', 'aptos', 'messidor', 'mbrset', 'papila']
        transform: torchvision.transforms.Compose object
        batch_size: int, batch size for dataloader
        
    Returns:
        LightningDataModule instance
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Choose from {list(DATASET_CONFIG.keys())}"
        )
    
    config = DATASET_CONFIG[dataset_name]
    return config["module"](
        root=config["path"],
        transform=transform,
        batch_size=batch_size,
    )


def get_class_weights(dataset_name, compute_weights=True):
    """
    Get class weights for the specified dataset.
    
    Args:
        dataset_name: str, one of ['idrid', 'aptos', 'messidor', 'mbrset', 'papila']
        compute_weights: bool, whether to compute weights (default True)
        
    Returns:
        torch.Tensor of class weights, or None if dataset has no weights or compute_weights=False
    """
    if not compute_weights:
        return None
    
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Choose from {list(DATASET_CONFIG.keys())}"
        )
    
    config = DATASET_CONFIG[dataset_name]
    weights_fn = config["weights_fn"]
    
    if weights_fn is None:
        return None
    
    try:
        weights = weights_fn(config["path"])
        return torch.tensor(weights, dtype=torch.float32)
    except Exception as e:
        print(f"[!] Failed to compute class weights for {dataset_name}: {e}")
        return None


def setup_dataloader_and_weights(dataset_name, transform, batch_size=BATCH_SIZE, compute_weights=True):
    """
    Convenience function to get both dataloader and class weights in one call.
    
    Args:
        dataset_name: str, dataset name
        transform: torchvision.transforms.Compose
        batch_size: int
        compute_weights: bool
        
    Returns:
        tuple (dataloader, class_weights)
    """
    dataloader = get_dataloader(dataset_name, transform, batch_size)
    class_weights = get_class_weights(dataset_name, compute_weights)
    return dataloader, class_weights
