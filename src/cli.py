
import typer, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import cv2, numpy as np

from src.datasets import LabelDataset, TestImageDataset, collate_fn, load_label_maps
from src.model import get_model_instance, save_checkpoint, load_checkpoint
from src.engine import fit, evaluate_map
from src.infer import save_predictions_to_csv

app = typer.Typer()

@app.command()
def train(
    train_img_dir: str,
    train_label_dir: str,
    val_img_dir: str,
    val_label_dir: str,
    label_maps: str,
    epochs: int = 10,
    batch_size: int = 4,
    accumulation_steps: int = 1,
    lr: float = 2e-4,
    weight_decay: float = 5e-2,
    ckpt_out: str = "best.pth",
):
    id_to_index, index_to_name = load_label_maps(label_maps)
    num_classes = len(index_to_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_ds = LabelDataset(train_img_dir, train_label_dir, transform=None)
    val_ds   = LabelDataset(val_img_dir,   val_label_dir,   transform=None)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = get_model_instance(num_classes=num_classes, pretrained=True).to(device)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=6, gamma=0.5)

    best_map = fit(model, train_dl, val_dl, optim, scheduler, device=device,
                   epochs=epochs, accumulation_steps=accumulation_steps)
    Path(ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optim, scheduler, epochs, ckpt_out)
    typer.echo(f"Saved checkpoint to {ckpt_out} (best mAP={best_map:.4f})")

@app.command()
def eval(
    ckpt: str,
    val_img_dir: str,
    val_label_dir: str,
    label_maps: str,
    batch_size: int = 4,
):
    id_to_index, index_to_name = load_label_maps(label_maps)
    num_classes = len(index_to_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    val_ds = LabelDataset(val_img_dir, val_label_dir, transform=None)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = get_model_instance(num_classes=num_classes, pretrained=False).to(device)
    load_checkpoint(model, ckpt)
    scores = evaluate_map(model, val_dl, device=device)
    typer.echo(scores)

@app.command()
def infer_csv(
    ckpt: str,
    test_img_dir: str,
    label_maps: str,
    out_csv: str = 'submission.csv',
    score_threshold: float = 0.5
):
    id_to_index, index_to_name = load_label_maps(label_maps)
    num_classes = len(index_to_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = TestImageDataset(test_img_dir, transform=None)
    model = get_model_instance(num_classes=num_classes, pretrained=False).to(device)
    load_checkpoint(model, ckpt)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    save_predictions_to_csv(ds, model, id_to_index, device, out_csv, score_threshold=score_threshold)
    typer.echo(f"Saved predictions to {out_csv}")

if __name__ == "__main__":
    app()
