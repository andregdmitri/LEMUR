import argparse
import torch
import torch.nn as nn
from thop import profile
import torch.nn.quantized.dynamic as nnqd

from train.train_retfound import RETFoundTask

try:
    from torch.utils.flop_counter import FlopCounterMode
    HAS_FLOP_COUNTER = True
except ImportError:
    HAS_FLOP_COUNTER = False

from models.vmamba_backbone import VisualMamba
from train.head import VMambaHeadTask
from train.train import MODEL_REGISTRY
from train.prune import apply_structured_pruning, apply_dynamic_quantization
from config.constants import IMG_SIZE, IN_CHANS, VMAMBA_EMBED_DIM, VMAMBA_DEPTH, PATCH_SIZE, NUM_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """
    Safely count parameters, extracting them from Quantized Packed layers 
    and counting actual non-zero values for pruned structural sparsity.
    """
    total_params = 0
    nonzero_params = 0
    
    for name, module in model.named_modules():
        # Handle PyTorch dynamic quantized linear layers
        if hasattr(module, '_packed_params'):
            try:
                weight, bias = module._packed_params._weight_bias()
                total_params += weight.numel()
                # Quantized tensors need to be converted to int_repr to count zeros
                if weight.is_quantized:
                    nonzero_params += torch.count_nonzero(weight.int_repr()).item()
                else:
                    nonzero_params += torch.count_nonzero(weight).item()
                    
                if bias is not None:
                    total_params += bias.numel()
                    nonzero_params += torch.count_nonzero(bias).item()
            except Exception:
                pass
        else:
            # Handle standard layers. recurse=False prevents double-counting.
            for param in module.parameters(recurse=False):
                total_params += param.numel()
                nonzero_params += torch.count_nonzero(param).item()
                
    return total_params, nonzero_params

# Custom THOP rule for Dynamic Quantized Linear
def count_quantized_linear(m, x, y):
    # MACs = Output Elements * Input Features
    macs = y.numel() * x[0].size(-1)
    m.total_ops += torch.DoubleTensor([int(macs)])


def verify_model(ckpt_path, model_name, is_pruned, is_quantized, prune_amount=0.2, prune_n=2, prune_dim=0):
    print(f"--- Initializing Model: {model_name} ---")
    
    # 1. Instantiate the correct architecture
    if model_name == "vmamba_head":
        # Phase II split architecture
        backbone = VisualMamba(
            img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS,
            embed_dim=VMAMBA_EMBED_DIM, depth=VMAMBA_DEPTH, 
            learning_rate=0.0, mask_ratio=0.0, use_cls_token=False,
        )
        model = VMambaHeadTask(backbone, lr=1e-4)
    elif model_name == "retfound":
        model = RETFoundTask.load_from_checkpoint(
            args.ckpt,
            strict=False,           
            mode="finetune",
            class_weights=None
        ).to("cpu")
    elif model_name in MODEL_REGISTRY:
        # Standard lightweight architectures
        model_cls = MODEL_REGISTRY[model_name]
        model = model_cls(
            num_classes=NUM_CLASSES,
            pretrained=False,
            lr=1e-4,
            class_weights=None
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from vmamba_head or {list(MODEL_REGISTRY.keys())}")

    # 2. Apply transforms FIRST to mutate the architecture to match the checkpoint
    if is_pruned:
        print("[*] Applying structured pruning scaffold...")
        if model_name == "vmamba_head":
            model.backbone = apply_structured_pruning(model.backbone, amount=prune_amount, n=prune_n, dim=prune_dim)
            model.head = apply_structured_pruning(model.head, amount=prune_amount, n=prune_n, dim=prune_dim)
        else:
            model = apply_structured_pruning(model, amount=prune_amount, n=prune_n, dim=prune_dim)
    
    if is_quantized:
        print("[*] Applying dynamic quantization scaffold...")
        if model_name == "vmamba_head":
            model.backbone = apply_dynamic_quantization(model.backbone)
            model.head = apply_dynamic_quantization(model.head)
        else:
            model = apply_dynamic_quantization(model)
        
    # 3. Load Weights
    print(f"[*] Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Handle both the custom VMamba split dict and standard Lightning module state_dicts
    if model_name == "vmamba_head" and "backbone" in ckpt and "head" in ckpt:
        full_state = {}
        for k, v in ckpt["backbone"].items(): full_state[f"backbone.{k}"] = v
        for k, v in ckpt["head"].items(): full_state[f"head.{k}"] = v
        model.load_state_dict(full_state, strict=False)
    else:
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    print(f"[✓] Successfully loaded model!\n")
    
    # 4. Parameters and Sparsity
    total, nonzero = count_parameters(model)
    sparsity = 100 * (1 - nonzero / total) if total > 0 else 0
    print("=== PARAMETER COUNTS ===")
    print(f"Total Parameters (Dense):  {total:,}")
    print(f"Non-Zero Parameters:       {nonzero:,}")
    print(f"Effective Sparsity:        {sparsity:.2f}%\n")
    
    # 5. FLOPs / MACs Verification
    print("=== COMPUTATION METRICS (MACs/FLOPs) ===")
    dummy_input = torch.randn(1, IN_CHANS, IMG_SIZE, IMG_SIZE)
    custom_ops = {nnqd.Linear: count_quantized_linear} if is_quantized else {}

    try:
        # THOP computes Dense MACs
        macs, params = profile(model, inputs=(dummy_input,), custom_ops=custom_ops, verbose=False)
        print(f"THOP Total MACs (Dense):   {macs / 1e9:.3f} GMACs")
        
        # Calculate theoretical savings if running on sparse-aware hardware
        if is_pruned:
            eff_macs = macs * (nonzero / total)
            print(f"Theoretical Sparse MACs:   {eff_macs / 1e9:.3f} GMACs")
    except Exception as e:
        print(f"[!] THOP analysis failed: {e}")

    if HAS_FLOP_COUNTER:
        try:
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                model(dummy_input)
            total_flops = flop_counter.get_total_flops()
            print(f"PyTorch 2.1+ ATen FLOPs:   {total_flops / 1e9:.3f} GFLOPS")
        except Exception as e:
             pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify FLOPs and Params for Pruned/Quantized Models")
    parser.add_argument("--model", type=str, required=True, 
                        help="Model architecture (mobilenet, efficientnet, unet, vmamba, or vmamba_head)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint (.pth or .ckpt)")
    parser.add_argument("--pruned", action="store_true", help="Set this flag if the checkpoint was pruned")
    parser.add_argument("--quantized", action="store_true", help="Set this flag if the checkpoint was quantized")
    parser.add_argument("--prune_amount", type=float, default=0.2, help="Must match training argument")
    parser.add_argument("--prune_n", type=int, default=2, help="Must match training argument")
    parser.add_argument("--prune_dim", type=int, default=0, help="Must match training argument")
    args = parser.parse_args()
    
    verify_model(args.ckpt, args.model, args.pruned, args.quantized, args.prune_amount, args.prune_n, args.prune_dim)