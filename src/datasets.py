
import os, json, cv2, torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class LabelDataset(Dataset):
    """YOLO txt (x_c,y_c,w,h normalized) -> COCO xyxy tensor로 변환"""
    def __init__(self, img_dir: str, label_dir: str, transform=None):
        self.img_paths = sorted([p for p in Path(img_dir).glob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg'}])
        self.label_dir = Path(label_dir)
        self.transform = transform

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes, labels = [], []
        label_path = self.label_dir / (img_path.stem + '.txt')
        if label_path.exists():
            for line in open(label_path, 'r', encoding='utf-8'):
                parts = line.strip().split()
                if len(parts) != 5: continue
                cls = int(parts[0]); x_c, y_c, bw, bh = map(float, parts[1:])
                x1 = (x_c - bw/2) * w; y1 = (y_c - bh/2) * h
                x2 = (x_c + bw/2) * w; y2 = (y_c + bh/2) * h
                boxes.append([x1, y1, x2, y2]); labels.append(cls)
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        img = torch.from_numpy(img).permute(2,0,1).float()/255.0
        target = {'boxes': boxes, 'labels': labels}
        if self.transform: img = self.transform(img)
        return img, target

class TestImageDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.root = image_dir; self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx: int):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.root, fname)).convert('RGB')
        if self.transform: img = self.transform(img)
        else: img = transforms.ToTensor()(img)
        return img, fname

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def load_label_maps(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    id_to_index = {int(k): int(v) for k, v in data['id_to_index'].items()}
    index_to_name = {int(k): v for k, v in data['index_to_name'].items()}
    return id_to_index, index_to_name
