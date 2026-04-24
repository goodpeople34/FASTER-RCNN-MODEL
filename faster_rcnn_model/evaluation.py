import torch
import torchvision
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from dataset import test_ds
from dataVisual import manga_collate_fn
from config import NUM_CLASSES, WIDTH, HEIGHT, CLASSES

num_classes = NUM_CLASSES
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes
)

model.load_state_dict(torch.load("model_epoch_40_512x512_2nd.pth", map_location=device))
model.eval()
model.to(device)

metric = MeanAveragePrecision(
    iou_type="bbox",
    box_format="xyxy",
    class_metrics=True
).to(device)

test_loader = DataLoader(
    test_ds,
    batch_size=2,
    shuffle=False,
    collate_fn=manga_collate_fn,
    num_workers=0
)

print("Starting evaluation...")
with torch.no_grad():
    for images, targets in tqdm(test_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]          # Important: list of tensors
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        # Convert outputs to CPU for torchmetrics (recommended)
        outputs = [{k: v.cpu() for k, v in out.items()} for out in outputs]
        targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

        metric.update(outputs, targets)

results = metric.compute()

print("\n" + "="*80)
print("                  mAP EVALUATION RESULTS")
print("="*80)
print(f"Image Size          : {WIDTH} x {HEIGHT}")
print(f"mAP (0.50:0.95)     : {results['map']:.4f}")
print(f"mAP @ 0.50          : {results['map_50']:.4f}")
print(f"mAP @ 0.75          : {results['map_75']:.4f}")
print("-" * 80)

print("Per-class Average Precision:")
foreground_classes = CLASSES[1:] if len(CLASSES) > 1 else ['bubble_text', 'square_text', 'sparky_text']

map_per_class = results.get('map_per_class', [])

for i, ap in enumerate(map_per_class):
    if i < len(foreground_classes):
        name = foreground_classes[i]
        gt_info = " (no GT in test set)" if ap == 0 else ""
        print(f" {name:18} : {ap:.4f}{gt_info}")
    else:
        print(f" Class_{i+1:2d}        : {ap:.4f}")

print("-" * 80)
print(f"Total test images   : {len(test_ds)}")
print("="*80)

print("\nAdditional Metrics:")
print(f"mAP small           : {results.get('map_small', 'N/A')}")
print(f"mAP medium          : {results.get('map_medium', 'N/A')}")
print(f"mAP large           : {results.get('map_large', 'N/A')}")