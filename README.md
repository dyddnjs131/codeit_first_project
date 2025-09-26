# First Project (Object Detection)

간단한 구조로 정리한 객체 탐지 레포입니다. **아키텍처 설명**과 **개선 성과/모델 성능**을 한 눈에 볼 수 있도록 구성했습니다.

## 개요
- **목표**: 소형 객체 탐지 정확도 개선 및 추론 속도 최적화
- **모델**: torchvision Faster R-CNN (ResNet-50 FPN v2)
- **데이터**: COCO 포맷(예시), 클래스/해상도는 `configs/data.yaml` 참고

## 아키텍처
```text
[Dataset/Transforms] -> [Model Builder] -> [Trainer] -> [Evaluator] -> [Reporter]
     (COCO)              (Faster R-CNN)    (epoch)         (mAP)         (CSV/plots)
```
- 데이터 파이프라인: `src/datasets.py` (COCO 로더/Collate)
- 모델 구성: `src/model.py`
- 학습 엔진: `src/engine.py`
- 평가/추론: `src/engine.py::evaluate`, `src/infer.py`

## 개선 포인트 & 성과
1. 라벨 정합성 확보(YOLO→COCO 변환 검증) → 학습 안정화
2. 학습 라벨 위치 및 이미지 매칭

**최종 성능**
- mAP@0.50: `0737 → 0.711`
- mAP@0.50:.95: `0.977 → 0.893`


# 시작 코드
pip install -r requirements.txt

# 학습
python -m src.cli train \
  --train-img-dir /path/train_images \
  --train-label-dir /path/labels/train \
  --val-img-dir /path/images/val \
  --val-label-dir /path/labels/val \
  --label-maps /path/label_maps.txt \
  --epochs 20 --batch-size 4 --accumulation-steps 8 \
  --ckpt-out runs/best.pth

# 평가
python -m src.cli eval \
  --ckpt runs/best.pth \
  --val-img-dir /path/images/val \
  --val-label-dir /path/labels/val \
  --label-maps /path/label_maps.txt

# 추론 CSV
python -m src.cli infer_csv \
  --ckpt runs/best.pth \
  --test-img-dir /path/test_images \
  --label-maps /path/label_maps.txt \
  --out-csv submission.csv

## 구조
```text
first-project/
├─ configs/           # 설정
├─ notebooks/         # 원본 노트북(백업)
├─ scripts/           # 변환/시각화 도구(옵션)
├─ src/               # 최소 모듈 (datasets/model/engine/infer/cli)
└─ /                  # 데이터 text

```

- `label_maps.txt`는 JSON: `{"id_to_index": {...}, "index_to_name": {...}}`
- 키는 자동으로 int로 캐스팅됩니다.
