import torch
import torch.nn.utils.prune as prune


def apply_structured_pruning(model, amount=0.2, n=2, dim=0):
    """Aplica pruning estruturado nas camadas conv e linear de forma simples."""
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            try:
                prune.ln_structured(module, name='weight', amount=amount, n=n, dim=dim)
                prune.remove(module, 'weight')
            except Exception:
                pass
    return model


def apply_dynamic_quantization(model, dtype=torch.qint8):
    """Quantização dinâmica (aplicável em CPU em PyTorch)"""
    qmodel = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=dtype
    )
    return qmodel


def benchmark_model(model, dataloader, device='cuda'):
    model.eval()
    model.to(device)
    total = 0.0
    n = 0
    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(device)
            logits = model(x)
            total += logits.shape[0]
            n += 1
            if n >= 10:
                break
    return total
