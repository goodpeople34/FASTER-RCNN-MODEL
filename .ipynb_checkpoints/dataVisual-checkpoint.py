import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.utils.data import DataLoader
from dataset import TRAIN_TXT,IMAGE_DIR,ANNOT_DIR,CLASSES, mangaDataset


class VOCvisualizer:
    def __init__(self,class_name):
        self.class_name = class_name

    def visualize_batch(self, data_loader, num_images):
        fig, ax = plt.subplots(1, num_images, figsize = (5 * num_images, 6))
        if num_images == 1:
            ax = [ax]
        else:
            ax = ax.ravel()

        data_loader_iter = iter(data_loader)
        images, targets = next(data_loader_iter)

        images = images[:num_images]
        targets = targets[:num_images]

        for idx in range(num_images):
            image = images[idx].permute(1,2,0).cpu().numpy()
            target = targets[idx]
            boxes = target["boxes"].cpu().numpy()
            labels = target["labels"].cpu().numpy()

            ax[idx].imshow(image)
            ax[idx].axis('off')

            for box, label in zip(boxes, labels):
                xmin, ymin, xmax, ymax = box

                width, height = xmax - xmin, ymax - ymin
                rect = patches.Rectangle(
                    (xmin, ymin),
                    width,
                    height,
                    linewidth = 2,
                    edgecolor = 'red',
                    facecolor = 'none',
                )

                ax[idx].add_patch(rect)

                class_names = self.class_name[labels] if label < len(self.class_name) else f"class {label}"
                ax[idx].text(
                    xmin,
                    ymin - 10,
                    class_names,
                    color = "red",
                    fontsize = 12,
                    bbox = dict(facecolor="yellow", alpha=0.5, edgecolor="red"),
                )
        plt.tight_layout()
        plt.show()

def manga_collate_fn(batch):
    """
    Collate function for batching images and targets.
    Stacks images into a batch tensor, keeps targets as a list of dicts.
    This prevents stacking errors for tensors of different sizes (e.g., different number of boxes per image).
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

train_ds = mangaDataset(
    annot_dir=ANNOT_DIR,
    image_dir=IMAGE_DIR,
    split_file=TRAIN_TXT,
    width=512,
    height=512,
    classes=CLASSES,
    transforms=None
)

train_data_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=manga_collate_fn, num_workers=0)
visualizer = VOCvisualizer(class_name=CLASSES)
visualizer.visualize_batch(train_data_loader, num_images=2)