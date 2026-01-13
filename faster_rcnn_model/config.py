import torchvision

DATASET_ID = 128
TRAIN_SIZE = 90
TRAIN_FILE_PATH = 'train_id.txt'
TEST_FILE_PATH = 'test_id.txt'


ANNOT_DIR = 'ANNOTATION FOLDER DIRECTORY'
IMAGE_DIR = 'IMAGE FOLDER DIRECTORY'
TRAIN_TXT = 'TRAIN.TXT FILE LOCATION'
CLASSES = ['background','bubble_text']
NUM_CLASSES = 2

MODEL = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
NUM_EPOCHS = 10

SAVE_MODEL_EPOCHS = 2