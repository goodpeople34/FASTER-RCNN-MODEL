import torch
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.utils.data import DataLoader
from dataset import train_ds
from config import CLASSES
from dataVisual import manga_collate_fn


class VOCCounter:
    def __init__(self, class_names):
        self.class_names = class_names

    def count_dataset(self, data_loader):
        total_counts = Counter()
        
        print(f"Counting labels in {len(data_loader)} batches...")
        
        for images, targets in data_loader:
            for target in targets:
                labels = target["labels"].cpu().numpy()
                
                for label in labels:
                    label_idx = int(label)
                    # Use class name if available, otherwise use the index
                    name = (self.class_names[label_idx] 
                            if label_idx < len(self.class_names) 
                            else f"class {label_idx}")
                    total_counts[name] += 1
                    
        return dict(total_counts)

# Implementation using your existing objects
train_data_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=manga_collate_fn, num_workers=0)
counter = VOCCounter(class_names=CLASSES)
label_totals = counter.count_dataset(train_data_loader)

# Print results
print("\n--- Final Dataset Summary ---")
for cls, count in label_totals.items():
    print(f"{cls}: {count} instances")


