import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np


ANNOT_DIR = '/home/moel/dataset labeled/LabeledVoc'
IMAGE_DIR = '/home/moel/dataset labeled/LabeledJpg'
TRAIN_TXT = '/home/moel/coding/FasterRCNN/train_id.txt'
CLASSES = ['background','bubble_text']
NUM_CLASSES = 1

class mangaDataset(Dataset):
    def __init__(self, annot_dir,image_dir, split_file, width, height, classes, transforms=None):
        self.transforms = transforms
        self.width = width
        self.height = height
        self.classes = classes
        self.annot_dir =annot_dir
        self.image_dir = image_dir

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

        if len(self.image_ids) == 0:
            raise ValueError(f"No image IDs found in {split_file}")

        print(f"Loaded {len(self.image_ids)} images from {os.path.basename(split_file)}")


    def __getitem__(self,idx):
        image_id = self.image_ids[idx]
        # image_path = os.path.join(self.image_dir, image_id)

        image_name = f"{image_id}.jpg"
        image_path = os.path.join(self.image_dir, image_name)

        xml_name = f"{image_id}.xml"
        xml_path = os.path.join(self.annot_dir, xml_name)

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        boxes=[]
        labels=[]
        tree=ET.parse(xml_path)
        root=tree.getroot()

        image_width = image.shape[1]
        image_height = image.shape[0]

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Annotation not found: {xml_path}")

        for member in root.findall('object'):
            name = member.find('name').text
            if name in self.classes:
                label = self.classes.index(name)  # Labels start from 0
            else:
                continue

            xmin = int(member.find('bndbox').find('xmin').text)

            xmax = int(member.find('bndbox').find('xmax').text)
            
            ymin = int(member.find('bndbox').find('ymin').text)

            ymax = int(member.find('bndbox').find('ymax').text)

            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = (boxes[:,3]-boxes[:,1]) * (boxes[:,2] - boxes[:,0])

        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image = image_resized, bboxes = target['boxes'], labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        else:
            image_resized = cv2.resize(image, (self.width, self.height))
            image_resized = image_resized.astype(np.float32) / 255.0
            image_resized = torch.from_numpy(image_resized.transpose(2, 0, 1))  # to (C,H,W)

        return image_resized, target

    def __len__(self):
        return len(self.image_ids)


# train_ds = mangaDataset(
#     annot_dir=ANNOT_DIR,
#     image_dir=IMAGE_DIR,
#     split_file=TRAIN_TXT,
#     width=512,
#     height=512,
#     classes=CLASSES,
#     transforms=None
# )   

xml_path = os.path.join(ANNOT_DIR, '1.xml')
print(xml_path)

with open(TRAIN_TXT, 'r') as f:
            image_ids = [line.strip() for line in f if line.strip()]

image_id = image_ids[1]
print(image_id)