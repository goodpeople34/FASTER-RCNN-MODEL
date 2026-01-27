import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
import torch
import torchvision
from PIL import Image
from config import TEST_FILE_PATH, IMAGE_DIR
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 2
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load("model_epoch50.pth",map_location=torch.device('cpu')))
model.eval()
model.to(device)

fig, ax = plt.subplots(1,1,figsize=(10, 6))
# ax = ax.ravel()
ax =[ax]

transform = transforms.ToTensor()

testId = TEST_FILE_PATH

split_file = testId

if not os.path.exists(split_file):
    raise FileNotFoundError(f"Split file not found: {split_file}")

with open(split_file, 'r') as f:
    image_ids = [line.strip() for line in f if line.strip()]

if len(image_ids) == 0:
    raise ValueError(f"No image IDs found in {split_file}")

print(f"Loaded {len(image_ids)} images from {os.path.basename(split_file)}")

for idx in range(1):


    # image_id = random.choice(image_ids)
    # image_name = f"{image_id}.jpg"
    # image_path = os.path.join(IMAGE_DIR, image_name)
    # print("image path :", image_path)

    image_id = "002"
    image_path = "/home/moel/Pictures/manga/Boku no Kanojo wa Dekkawai/extracted/Chapter 3/002.jpg"
    print("image path :", image_path)

    image_pil= Image.open(image_path)
    image = Image.open(image_path)
    image = transform(image)
    image_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)[0]
    print(prediction)

    image_np = image.cpu().permute(1, 2, 0).numpy()

    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    ax[idx].imshow(image_np,cmap='gray')
    ax[idx].axis("off")
    crop_idx = 0

    # for box, label, score in zip(boxes, labels, scores):
    #     if score >= 0.8:
    #         xmin, ymin, xmax, ymax = box

    #         xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

    #         cropped_img = image_pil.crop((xmin, ymin, xmax, ymax))

    #         # Optional: save cropped image
    #         save_path = f"crop_{image_id}_{crop_idx}.jpg"
    #         cropped_img.save(save_path)
    #         crop_idx += 1

    #         # Draw rectangle
    #         width, height = xmax - xmin, ymax - ymin
    #         rect = patches.Rectangle(
    #             (xmin, ymin),
    #             width,
    #             height,
    #             linewidth=2,
    #             edgecolor="red",
    #             facecolor="none",
    #         )
    #         ax[idx].add_patch(rect)

    #         ax[idx].text(
    #             xmin,
    #             ymin - 10,
    #             f"Class {label} ({score:.2f})",
    #             color="red",
    #             fontsize=10,
    #             bbox=dict(facecolor="yellow", alpha=0.5),
    #         )


    for box, label, score in zip(boxes, labels, scores):
        if score >= 0.8:
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            rect = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax[idx].add_patch(rect)
            ax[idx].text(
                xmin + 8,
                ymin - 20,
                f"Class {label}",
                color="red",
                fontsize=5,
                bbox=dict(facecolor="yellow", alpha=0.5, edgecolor="red"),
            )
            
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    pil_img = Image.open(buf)

    qt_img = ImageQt(pil_img)
    # pil_img.show()

pil_img.show()
