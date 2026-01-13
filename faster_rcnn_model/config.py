import torchvision

DATASET_ID = 128
TRAIN_SIZE = 90
TRAIN_FILE_PATH = 'train_id.txt'
TEST_FILE_PATH = 'test_id.txt'


ANNOT_DIR = '/home/moel/dataset labeled/LabeledVoc'
IMAGE_DIR = '/home/moel/dataset labeled/LabeledJpg'
TRAIN_TXT = '/home/moel/coding/FasterRCNN/train_id.txt'
CLASSES = ['background','bubble_text']
NUM_CLASSES = 2

MODEL = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
NUM_EPOCHS = 10

SAVE_MODEL_EPOCHS = 2