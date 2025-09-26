
import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model_instance(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Faster R-CNN ResNet50 FPN v2, 클래스 수만 교체"""
    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT' if pretrained else None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def save_checkpoint(model: nn.Module, optimizer, scheduler, epoch: int, path: str):
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict() if optimizer else None,
                'scheduler': scheduler.state_dict() if scheduler else None,
                'epoch': epoch}, path)

def load_checkpoint(model: nn.Module, path: str, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer and ckpt.get('optimizer'):
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler and ckpt.get('scheduler'):
        scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt.get('epoch', 0)
