import torchvision

# dataset configuration
DATASET_ID = 407
TRAIN_SIZE = 325
# WIDTH AND HEIGHT PARAMETER
# 512x512
# 356x534
# 460x460

#invert image True or False
INVERT = True

# image resize configuration
WIDTH = 460
HEIGHT = 460

# dataset for 2 classes of label
# ANNOT_DIR = '/home/moel/dataset labeled1/LabeledVoc'
# IMAGE_DIR = '/home/moel/dataset labeled1/LabeledJpg'
# TRAIN_TXT = '/home/moel/coding/FasterRCNN1/train_id.txt'
# CLASSES = ['background','bubble_text']
# NUM_CLASSES = 2

# dataset for 4 classes of label 
ANNOT_DIR = '/home/moel/dataset labeled2/datasetVOC'
IMAGE_DIR = '/home/moel/dataset labeled2/datasetJPG'
TRAIN_TXT = '/home/moel/coding/FasterRCNN1/train_id.txt'
TEST_TXT = '/home/moel/coding/FasterRCNN1/test_id.txt'
CLASSES = ['background','bubble_text','square_text','sparky_text']
NUM_CLASSES = 4

# model backbone configuration
MODEL = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
NUM_EPOCHS = 40

SAVE_MODEL_EPOCHS = 10




