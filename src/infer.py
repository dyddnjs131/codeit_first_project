
import csv, torch
import numpy as np
import matplotlib.pyplot as plt

def plot_prediction(image, prediction, threshold=0.5, fontproperties=None):
    img = image.permute(1,2,0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img); ax.axis('off')
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    for box, score, label in zip(boxes, scores, labels):
        if score < threshold: continue
        x1, y1, x2, y2 = map(int, box)
        rect = plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, max(y1-3,0), f"{int(label)}:{score:.2f}", fontsize=9,
                color='blue', fontproperties=fontproperties,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def save_predictions_to_csv(dataset, model, id_to_index, device, out_csv_path, score_threshold=0.5):
    results = []
    annotation_id = 1
    class_mapping = {v: k for k, v in id_to_index.items()}  # model label -> original category_id
    for idx in range(len(dataset)):
        item = dataset[idx]
        if isinstance(item, tuple) and len(item) == 2:
            img, fname = item
        else:
            img, fname = item, f"{idx}.png"
        img = img.unsqueeze(0) if isinstance(img, torch.Tensor) and img.ndim == 3 else img
        pred = model(img.to(device))[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        image_id = idx
        for box, label, score in zip(boxes, labels, scores):
            if score < score_threshold: continue
            x1, y1, x2, y2 = box
            row = [annotation_id, image_id, class_mapping.get(int(label), -1),
                   int(x1), int(y1), int(x2-x1), int(y2-y1), round(float(score),2)]
            results.append(row); annotation_id += 1
    with open(out_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['annotation_id','image_id','category_id','bbox_x','bbox_y','bbox_w','bbox_h','score'])
        writer.writerows(results)
