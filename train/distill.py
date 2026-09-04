import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils.transforms import build_train_transform, eval_transform
from config.constants import *
from models.retfound import RETFoundBackbone
from models.vmamba_backbone import VisualMamba
from models.tinyvit import TinyViTStudent
from models.dist import DistillationModule
from dataloader import get_dataloader

# -----------------------------------------------------------
#  Helpers: Model Builders
# -----------------------------------------------------------

def build_student(args):
    """Initializes the VMamba student backbone."""
    return VisualMamba(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        embed_dim=VMAMBA_EMBED_DIM,
        depth=VMAMBA_DEPTH,
        learning_rate=args.lr or LR,
        mask_ratio=args.mask_ratio or MASK_RATIO,
        use_cls_token=False
    )

def build_tinyvit_student(args):
    """Initializes the TinyViT student backbone (encoder-only, no masking)."""
    return TinyViTStudent(
        model_name=TINYVIT_MODEL,
        pretrained=False,
    )

def build_teacher():
    """Initializes and freezes the RETFound teacher."""
    path = os.path.join(CHECKPOINT_DIR, "RETFound_cfp_weights.pth")
    teacher = RETFoundBackbone(checkpoint_path=path)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher

def run_distillation(args):
    seed = args.seed or SEED
    pl.seed_everything(seed)

    student_type = getattr(args, "student", "vmamba")
    print(f"\n=== PHASE I: {student_type.upper()} DISTILLATION (teacher=RETFound) ===")

    # 1. Models
    teacher = build_teacher()
    if student_type == "tinyvit":
        student = build_tinyvit_student(args)
    else:
        student = build_student(args)

    # Student embed dim (VMamba: VMAMBA_EMBED_DIM, TinyViT: num_features)
    student_embed_dim = getattr(student, "embed_dim", VMAMBA_EMBED_DIM)

    # Project student output dim to teacher dim (TEACHER_EMBED_DIM)
    # This is often needed if the student is smaller than the teacher.
    projector = nn.Sequential(
        nn.Linear(student_embed_dim, 2 * student_embed_dim),
        nn.GELU(),
        nn.Linear(2 * student_embed_dim, TEACHER_EMBED_DIM),
    )

    # 2. Data
    train_tfm = build_train_transform(getattr(args, "augmentation", "default"), IMG_SIZE)
    val_tfm = eval_transform(IMG_SIZE)
    dm = get_dataloader(args.dataset, (train_tfm, val_tfm), batch_size=BATCH_SIZE)
    dm.setup(stage="fit")

    # 3. Distillation Wrapper
    model = DistillationModule(
        teacher=teacher,
        student=student,
        projector=projector,
        lr=args.lr or LR,
    )

    # 4. Callbacks & Trainer
    ckpt_cb = ModelCheckpoint(
        monitor="train/distill_loss",
        mode="min",
        save_top_k=1,
        filename=f"best_distillation_{seed}_{args.dataset}"
    )
    
    early_cb = EarlyStopping(
        monitor="val/distill_loss",
        patience=PATIENCE,
        mode="min"
    )

    import time
    run_name = f"distill_{student_type}_{args.dataset}_{int(time.time())}"
    trainer = pl.Trainer(
        max_epochs=args.dist_epochs or DIST_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        logger=WandbLogger(project="distillation", name=run_name),
        callbacks=[ckpt_cb, early_cb],
        log_every_n_steps=5
    )

    # 5. Run Distillation
    trainer.fit(model, datamodule=dm)

    # 6. Final Extraction & Saving
    best_path = ckpt_cb.best_model_path

    # Load best weights into the model and compute metrics on train & val
    try:
        # load checkpoint state dict
        best_ckpt = torch.load(best_path, map_location="cpu")
        state = best_ckpt.get("state_dict", best_ckpt)
        model.load_state_dict(state, strict=False)

        # Ensure datamodule has loaders
        dm.setup(stage="fit")
        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()

        val_metrics = trainer.validate(model, dataloaders=val_dl)
        train_metrics = trainer.validate(model, dataloaders=train_dl)
    except Exception:
        val_metrics = []
        train_metrics = []

    # Save artifacts & results
    save_backbone_and_projector(best_path, student_type=student_type)

    # Append results to CSV
    try:
        from utils.results import append_result
        row = {
            "timestamp": int(time.time()),
            "mode": "distill",
            "dataset": args.dataset,
            "model_path": best_path,
            "run_name": run_name,
            "seed": args.seed,
            "monitor": ckpt_cb.monitor if hasattr(ckpt_cb, "monitor") else None,
            "monitor_value": float(ckpt_cb.best_model_score) if getattr(ckpt_cb, "best_model_score", None) is not None else None,
            "train_metrics": train_metrics[0] if train_metrics else {},
            "val_metrics": val_metrics[0] if val_metrics else {},
        }
        append_result(row)
    except Exception:
        pass


def save_backbone_and_projector(best_ckpt_path, student_type="vmamba"):
    """Extracts student and projector weights from the unified Lightning checkpoint."""
    print(f"\n[i] Extracting weights from: {best_ckpt_path}")
    
    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]

    # Extracting by prefixing logic
    student_sd = {k.replace("student.", ""): v for k, v in sd.items() if k.startswith("student.")}
    projector_sd = {k.replace("projector.", ""): v for k, v in sd.items() if k.startswith("projector.")}

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    student_path = os.path.join(CHECKPOINT_DIR, f"{student_type}_distilled_student.pth")
    projector_path = os.path.join(CHECKPOINT_DIR, f"{student_type}_distilled_projector.pth")

    # Saving in the format required by the 'Head' training Phase II
    torch.save({"backbone": student_sd}, student_path)
    torch.save({"projector": projector_sd}, projector_path)

    print(f"[✓] Saved Student Backbone: {student_path}")
    print(f"[✓] Saved Projector:        {projector_path}\n")