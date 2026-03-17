from ultralytics import YOLO
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset/dataset.yaml"
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
PRETRAINED = ROOT / "pretrained/yolo26n.pt"


def copy_best_model():
    src_model = RUNS / "coin_detector/weights/best.pt"
    dst_model = MODELS / "coin_detector.pt"
    shutil.copy(src_model, dst_model)


def train():
    model = YOLO(PRETRAINED)
    model.train(
        data=DATASET,
        epochs=50,
        imgsz=640,
        batch=8,
        device="cpu",
        project=RUNS,
        name="coin_detector",
        exist_ok=True,
        workers=2,
    )

    print("Training finished!")


if __name__ == "__main__":
    train()
    copy_best_model()


# 50 epochs completed in 1.638 hours.
# Optimizer stripped from /home/maksim/coins-object-detection-yolo/runs/coin_detector/weights/last.pt, 5.4MB
# Optimizer stripped from /home/maksim/coins-object-detection-yolo/runs/coin_detector/weights/best.pt, 5.4MB

# Validating /home/maksim/coins-object-detection-yolo/runs/coin_detector/weights/best.pt...
# Ultralytics 8.4.22 🚀 Python-3.12.3 torch-2.10.0+cu128 CPU (Intel Core i5-4440 3.10GHz)
# YOLO26n summary (fused): 122 layers, 2,375,421 parameters, 0 gradients, 5.2 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 25% ━━━───────── 1/4 8.9
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 50% ━━━━━━────── 2/4 6.0
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 75% ━━━━━━━━━─── 3/4 4.7
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 2.9s/it 11.7s
#                    all         60        121       0.91      0.891      0.955      0.945
#          10_uah_silver         28         28      0.889      0.893      0.929      0.918
#             1_uah_gold         25         34       0.95      0.941      0.992      0.988
#           1_uah_silver         36         59      0.892      0.839      0.943       0.93
# Speed: 1.7ms preprocess, 83.8ms inference, 0.0ms loss, 0.1ms postprocess per image
# Results saved to /home/maksim/coins-object-detection-yolo/runs/coin_detector
# Training finished!
