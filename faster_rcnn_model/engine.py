import torch
import torchvision
import cv2
from torchvision.transforms import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import train_ds
from dataVisual import train_data_loader

from config import MODEL, NUM_EPOCHS, SAVE_MODEL_EPOCHS, NUM_CLASSES



class MLModel:
    def __init__(self, model_name, num_classes, num_epoch):
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_epoch = num_epoch
        self.train_dataset = train_ds
        self.train_data_loader = train_data_loader

    def training(self):

        csv_path = "training_losses.csv"

        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "loss_total",
                    "loss_rpn_cls",
                    "loss_rpn_bbox",
                    "loss_roi_cls",
                    "loss_roi_bbox"
                ])

        model = self.model_name
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, self.num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )
        data_loader_iter = iter(self.train_data_loader)
        images, targets = next(data_loader_iter)

        num_epochs = self.num_epoch

        train_loss = []

        for epoch in range(num_epochs):
            print(f'starting the training of epoch {epoch+1}...')
            print('training...')

            model.train()
            rpn_cls_loss = 0.0
            rpn_box_loss = 0.0
            roi_cls_loss = 0.0
            roi_box_loss = 0.0
            total_loss = 0.0

            print(f'epoch {epoch}/{num_epochs} training')

            for images, targets in self.train_data_loader:
                images = list(image.to(device) for image in images)
                targets = [{k:v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)

                loss_objectness = loss_dict["loss_objectness"]
                loss_rpn_box = loss_dict["loss_rpn_box_reg"]
                loss_classifier = loss_dict["loss_classifier"]
                loss_box_reg = loss_dict["loss_box_reg"]

                losses = sum(loss for loss in loss_dict.values())

                rpn_cls_loss += loss_objectness.item()
                rpn_box_loss += loss_rpn_box.item()
                roi_cls_loss += loss_classifier.item()
                roi_box_loss += loss_box_reg.item()
                total_loss += losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            lr_scheduler.step()

            train_loss.append(total_loss)

            num_batches = len(self.train_data_loader)

            avg_total_loss = total_loss / num_batches
            avg_rpn_cls = rpn_cls_loss / num_batches
            avg_rpn_box = rpn_box_loss / num_batches

            print(
                f"Total loss: {avg_total_loss:.4f} | "
                f"RPN cls: {avg_rpn_cls:.4f} | "
                f"RPN bbox: {avg_rpn_box:.4f}"
            )

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    total_loss / num_batches,
                    rpn_cls_loss / num_batches,
                    rpn_box_loss / num_batches,
                    roi_cls_loss / num_batches,
                    roi_box_loss / num_batches
                ])

            if (epoch+1) % SAVE_MODEL_EPOCHS == 0:
                torch.save(model.state_dict(), f"model{epoch+1}.pth")
                print('SAVING MODEL COMPLETE...\n')


trainingModel = MLModel(MODEL,NUM_CLASSES,NUM_EPOCHS)
trainingModel.training()

