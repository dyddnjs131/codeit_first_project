
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def train_one_epoch(model, loader: DataLoader, optimizer, device='cuda', accumulation_steps=1, log_interval=50):
    model.train()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, (images, targets) in enumerate(loader, 1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        (loss/accumulation_steps).backward()
        running += float(loss.item())
        if step % accumulation_steps == 0 or step == len(loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % log_interval == 0:
            print(f"[train] {step:5d}/{len(loader)} loss {running/step:.4f}")
    return running / max(step, 1)

@torch.no_grad()
def evaluate_map(model, loader: DataLoader, device='cuda'):
    model.eval()
    metric = MeanAveragePrecision(iou_type='bbox')
    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds = model(images)
        gts = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
        metric.update([{k: v.to('cpu') for k, v in p.items()} for p in preds], gts)
    out = metric.compute()
    return {k: (float(v.item()) if hasattr(v, 'item') else float(v)) for k, v in out.items()}

def fit(model, train_loader, val_loader, optimizer, scheduler=None, device='cuda',
        epochs=10, accumulation_steps=1):
    best_map = -1.0; best_state = None
    for epoch in range(1, epochs+1):
        tloss = train_one_epoch(model, train_loader, optimizer, device=device,
                                accumulation_steps=accumulation_steps)
        if scheduler: scheduler.step()
        scores = evaluate_map(model, val_loader, device=device)
        cur_map = scores.get('map', 0.0)
        print(f"[epoch {epoch}] loss={tloss:.4f} mAP={cur_map:.4f}")
        if cur_map > best_map:
            best_map = cur_map
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return best_map
