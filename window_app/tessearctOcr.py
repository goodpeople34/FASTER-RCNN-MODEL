import pytesseract
from PIL import Image
from PIL.ImageQt import ImageQt
import torch
import torchvision
from torchvision.transforms import transforms
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import re

class CallModel:
    def _model(self, _img_path):
        self.extracted_text = []

        num_classes = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        model.load_state_dict(torch.load("second_model.pth",map_location=torch.device('cpu')))
        model.eval()
        model.to(device)

        matplotlib.use('agg')

        fig, ax = plt.subplots(figsize=(10, 6))

        transform = transforms.ToTensor()

        ax.clear()

        image_pil= Image.open(_img_path)
        image = Image.open(_img_path).convert("RGB")
        image = transform(image)
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(image_tensor)[0]

        image_np = image.cpu().permute(1, 2, 0).numpy()

        boxes = prediction["boxes"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()

        ax.imshow(image_np,cmap='gray')
        ax.axis("off")

        for box,label, score in zip(boxes,labels, scores):
            if score >= 0.8:
                self.words = []
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
                ax.add_patch(rect)
                ax.text(
                xmin + 8,
                ymin - 20,
                f"Class {label}",
                color="red",
                fontsize=5,
                bbox=dict(facecolor="yellow", alpha=0.5, edgecolor="red"),
                )

                xmin, ymin, xmax, ymax = map(int, box)
                cropped_img = image_pil.crop((xmin, ymin, xmax, ymax))
                self._tesseractModel(cropped_img)
                self.extracted_text.append(self.words)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight',pad_inches=0)
        buf.seek(0)

        final_pil = Image.open(buf)
        qt_image = ImageQt(final_pil)

        return qt_image, self.extracted_text

    def _tesseractModel(self, image_cropped):
            custom_config = r'--oem 3 --psm 6'
            extract_text = pytesseract.image_to_string(image_cropped, config=custom_config)
            clean_text = extract_text.strip() 
            
            clean_text = "".join([c if ord(c) < 128 else "" for c in clean_text])
            
            clean_text = re.sub(r'\n+', '\n', clean_text)
            clean_text = re.sub(r' +', ' ', clean_text)
            

            if clean_text:
                self.words.append(clean_text)
