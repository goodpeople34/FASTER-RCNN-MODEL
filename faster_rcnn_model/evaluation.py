import torch
import torchvision
from tqdm import tqdm
import numpy as np


def _iou_evluation(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])

    intersec_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    union_area = float(boxAArea + boxBArea - intersec_area)


    iou = intersec_area / union_area

    return iou


def iou_matrix(pred_boxes, gt_boxes):
    ious = torch.zeros((len(pred_boxes), len(gt_boxes)))

    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            ious[i, j] = box_iou(p, g)

    return ious


def _average_precission(recalls, precissions):

    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])


    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum(
        (recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]
    )

    return ap


def _mean_average_precission(pred_boxes,pred_scores, gt_boxes, iou_threshold):
    indices = np.argsort(-np.array(pred_scores))
    pred_boxes = [pred_boxes[i] for i in indices]

    matched_gt = [False] * len(gt_boxes)

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    for i, pred in enumerate(pred_boxes):
        best_iou = 0
        best_gt = -1

        for j, gt in enumerate(gt_boxes):
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = j
                
                
        if best_iou >= iou_threshold and not matched_gt[best_gt]:
            tp[i] = 1
            matched_gt[best_gt] = True
        else:
            fp[i] = 1




    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / max(len(gt_boxes), 1)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

    ap = average_precision(recalls, precisions)

    return ap


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(
    torch.load("model_epoch50.pth", map_location=device)
)
model.to(device)
model.eval()


def _evaluate_model(model, data_loader, iou_threshold=0.5):
    aps = []
    ious = []

    
    for images, targets in tqdm(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    outputs = model(images)


    for output, target in zip(outputs, targets):
            pred_boxes = output["boxes"].cpu().numpy()
            pred_scores = output["scores"].cpu().numpy()
            gt_boxes = target["boxes"].cpu().numpy()


            for gt in gt_boxes:
                if len(pred_boxes) == 0:
                    ious.append(0.0)
                    continue

                best_iou = max(
                    box_iou(pred, gt) for pred in pred_boxes
                )
                ious.append(best_iou)

                
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                ap = _mean_average_precission(
                    pred_boxes,
                    pred_scores,
                    gt_boxes,
                    iou_threshold=iou_threshold
                )
                aps.append(ap)


    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_ap = float(np.mean(aps)) if aps else 0.0

    return mean_iou, mean_ap

