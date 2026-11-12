"""Training pipeline: AdamW, LR scheduling, early stopping on validation macro-F1, checkpointing.

Run `python -m src.train` (use `--smoke` for a 2-batch sanity run).
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .dataset import TomatoLeafDataset, class_counts_from_csv
from .model import ModelConfig, build_model, model_summary
from .preprocessing import get_eval_transforms, get_train_transforms
from .utils import (
    DEFAULT_SEED,
    MODELS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SPLITS_DIR,
    ensure_dirs,
    environment_report,
    get_device,
    load_json,
    relative_to_project,
    save_json,
    seed_worker,
    set_seed,
)

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


def make_loaders(batch_size: int, num_workers: int, img_size: int, seed: int) -> dict[str, DataLoader]:
    gen = torch.Generator().manual_seed(seed)
    datasets = {
        "train": TomatoLeafDataset(SPLITS_DIR / "train.csv", get_train_transforms(img_size)),
        "val": TomatoLeafDataset(SPLITS_DIR / "val.csv", get_eval_transforms(img_size)),
    }
    common = dict(num_workers=num_workers, persistent_workers=num_workers > 0, worker_init_fn=seed_worker)
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, drop_last=False, generator=gen, **common),
        "val": DataLoader(datasets["val"], batch_size=batch_size * 2, shuffle=False, **common),
    }


def class_weights(counts: torch.Tensor) -> torch.Tensor:
    """Inverse-square-root frequency weights, normalised to mean 1 (mild imbalance correction)."""
    w = 1.0 / torch.sqrt(counts.float().clamp(min=1))
    return w / w.mean()


def run_epoch(model, loader, criterion, device, num_classes: int, optimizer=None, max_batches: int | None = None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    preds, targets = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        p = logits.argmax(1)
        correct += (p == y).sum().item()
        seen += x.size(0)
        preds.append(p.cpu())
        targets.append(y.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return {
        "loss": float(total_loss / max(seen, 1)),
        "acc": float(correct / max(seen, 1)),
        "macro_f1": float(f1_score(targets, preds, labels=list(range(num_classes)), average="macro", zero_division=0)),
    }


def plot_history(history: pd.DataFrame, path: Path, best_epoch: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    e = history["epoch"]
    axes[0].plot(e, history["train_loss"], "o-", label="train loss")
    axes[0].plot(e, history["val_loss"], "s-", label="val loss")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(e, history["train_acc"], "o-", label="train accuracy")
    axes[1].plot(e, history["val_acc"], "s-", label="val accuracy")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].set_ylim(top=1.0); axes[1].legend()
    axes[2].plot(e, history["val_macro_f1"], "s-", color="#B14C4C", label="val macro-F1")
    axes[2].plot(e, history["train_macro_f1"], "o-", color="#4C8C4A", alpha=0.6, label="train macro-F1")
    axes[2].set_title("Macro-F1"); axes[2].set_xlabel("epoch"); axes[2].set_ylim(top=1.0); axes[2].legend()
    for ax in axes:
        ax.axvline(best_epoch, color="gray", ls="--", lw=0.8)
        ax.grid(alpha=0.3)
    fig.suptitle("Training history (dashed line = best validation macro-F1 epoch)")
    fig.savefig(path)
    plt.close(fig)


def train(args: argparse.Namespace) -> dict:
    ensure_dirs()
    set_seed(args.seed)
    device = get_device(args.device)
    env = environment_report(device)
    print("Environment:", env)

    mapping = load_json(MODELS_DIR / "class_mapping.json")
    num_classes = mapping["num_classes"]
    config = ModelConfig(num_classes=num_classes, img_size=args.img_size, hidden_dim=args.hidden_dim,
                         dropout=args.dropout, trainable_stages=args.trainable_stages)
    model = build_model(config).to(device)
    summary = model_summary(model)
    print(summary)
    tag = "_smoke" if args.smoke else ""
    ckpt_path = MODELS_DIR / f"best_model{tag}.pth"
    (RESULTS_DIR / f"model_summary{tag}.txt").write_text(summary + "\n")

    loaders = make_loaders(args.batch_size, args.num_workers, args.img_size, args.seed)
    counts = class_counts_from_csv(SPLITS_DIR / "train.csv", num_classes)
    weights = class_weights(counts).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.backbone_lr}, {"params": head_params, "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    hparams = {k: v for k, v in vars(args).items()}
    hparams["class_weights"] = [round(float(w), 4) for w in weights.cpu()]
    hparams["train_class_counts"] = counts.tolist()
    hparams["optimizer"] = "AdamW"
    hparams["scheduler"] = "ReduceLROnPlateau(max, factor=0.5, patience=2) on val macro-F1"
    hparams["loss"] = f"CrossEntropyLoss(inverse-sqrt class weights, label_smoothing={args.label_smoothing})"

    history: list[dict] = []
    best_f1, best_epoch, best_state, epochs_without_improvement = -1.0, -1, None, 0
    max_batches = 2 if args.smoke else None
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, loaders["train"], criterion, device, num_classes, optimizer, max_batches)
        va = run_epoch(model, loaders["val"], criterion, device, num_classes, None, max_batches)
        scheduler.step(va["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"], "train_acc": tr["acc"], "train_macro_f1": tr["macro_f1"],
            "val_loss": va["loss"], "val_acc": va["acc"], "val_macro_f1": va["macro_f1"],
            "lr_backbone": optimizer.param_groups[0]["lr"], "lr_head": optimizer.param_groups[1]["lr"],
            "epoch_seconds": time.time() - t0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(RESULTS_DIR / f"training_history{tag}.csv", index=False)
        print(f"[{epoch:02d}/{args.epochs}] train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
              f"val loss {va['loss']:.4f} acc {va['acc']:.4f} macroF1 {va['macro_f1']:.4f} | "
              f"lr {row['lr_backbone']:.1e}/{row['lr_head']:.1e} | {row['epoch_seconds']:.0f}s", flush=True)

        if va["macro_f1"] > best_f1 + args.min_delta:
            best_f1, best_epoch, epochs_without_improvement = va["macro_f1"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model_state_dict": best_state, "config": config.to_dict(), "class_mapping": mapping,
                        "epoch": epoch, "val_metrics": va, "hyperparameters": hparams, "environment": env},
                       ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}: no val macro-F1 improvement for {args.patience} epochs")
                break

    total_seconds = time.time() - t_start
    hist_df = pd.DataFrame(history)
    plot_history(hist_df, PLOTS_DIR / f"training_curves{tag}.png", best_epoch)
    save_json(config.to_dict(), MODELS_DIR / f"model_config{tag}.json")
    summary_out = {
        "device": str(device), "environment": env, "seed": args.seed, "hyperparameters": hparams,
        "epochs_run": len(history), "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "best_val_acc": float(hist_df.loc[hist_df["epoch"] == best_epoch, "val_acc"].iloc[0]) if best_epoch > 0 else None,
        "total_training_seconds": total_seconds, "mean_epoch_seconds": float(hist_df["epoch_seconds"].mean()),
        "checkpoint": relative_to_project(ckpt_path), "smoke_run": args.smoke,
    }
    save_json(summary_out, RESULTS_DIR / f"training_summary{tag}.json")
    print(f"Done. Best val macro-F1 {best_f1:.4f} at epoch {best_epoch}; total {total_seconds/60:.1f} min")
    return summary_out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the tomato leaf CNN + MLP classifier")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--trainable-stages", nargs="*", default=["layer3", "layer4"])
    p.add_argument("--backbone-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default=None, help="force 'mps' or 'cpu'; default auto")
    p.add_argument("--smoke", action="store_true", help="run 2 batches per epoch to test the pipeline")
    return p.parse_args(argv)


if __name__ == "__main__":
    matplotlib.use("Agg")
    train(parse_args())
