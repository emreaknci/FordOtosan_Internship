import os


# Path to jsons
JSON_DIR = '../data/jsons'
if not os.path.exists(JSON_DIR):
    os.mkdir(JSON_DIR)

# Path to mask
MASK_DIR = '../data/masks'
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)

# Path to output images
IMAGE_OUT_DIR = '../data/masked_images'
if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(IMAGE_OUT_DIR)

# Path to original images
IMAGE_DIR = '../data/images'

# Path to model
MODEL_DIR = '../models/model.pt'

# Path to predict data
PREDICT_DIR = '../../train_data_1/predicts'
if not os.path.exists(PREDICT_DIR):
    os.mkdir(PREDICT_DIR)

PREDICT_MASK_DIR = '../../train_data_1/predict_masks'
if not os.path.exists(PREDICT_MASK_DIR):
    os.mkdir(PREDICT_MASK_DIR)

PREDICT_MASKED_IMAGES_DIR = '../../train_data_1/predict_masked_images'
if not os.path.exists(PREDICT_MASKED_IMAGES_DIR):
    os.mkdir(PREDICT_MASKED_IMAGES_DIR)

AUGMENTATION_DIR = '../../train_data_1/augmentations'
if not os.path.exists(AUGMENTATION_DIR):
    os.mkdir(AUGMENTATION_DIR)

AUGMENTATION_MASK_DIR = '../../train_data_1/augmentation_masks'
if not os.path.exists(AUGMENTATION_MASK_DIR):
    os.mkdir(AUGMENTATION_MASK_DIR)

AUGMENTATION_MASKED_IMAGES_DIR = '../../train_data_1/augmentation_masked_images'
if not os.path.exists(AUGMENTATION_MASKED_IMAGES_DIR):
    os.mkdir(AUGMENTATION_MASKED_IMAGES_DIR)

MODEL_AUG0 = '../../train_data_1/MODEL_AUG0'
if not os.path.exists(MODEL_AUG0):
    os.mkdir(MODEL_AUG0)

MODEL_AUG1 = '../../train_data_1/MODEL_AUG1'
if not os.path.exists(MODEL_AUG1):
    os.mkdir(MODEL_AUG1)

BESTMODEL_AUG0 = '../../train_data_1/BESTMODEL_AUG0'
if not os.path.exists(BESTMODEL_AUG0):
    os.mkdir(BESTMODEL_AUG0)

BESTMODEL_AUG1 = '../../train_data_1/BESTMODEL_AUG1'
if not os.path.exists(BESTMODEL_AUG1):
    os.mkdir(BESTMODEL_AUG1)

# In order to visualize masked-image(s), change "False" with "True"
VISUALIZE = False

# Bacth size
BATCH_SIZE = 4

# Input dimension
HEIGHT = 224
WIDTH = 224
INPUT_SHAPE = (HEIGHT, WIDTH)
# Number of class, for this task it is 2: Non-drivable area and Driviable area
N_CLASS = 2


# Path to train data
TRAIN_IMG_DIR = '../../train_data_1/images'
TRAIN_JSON_DIR = '../../train_data_1/jsons'
TRAIN_MASKED_IMAGES_DIR = '../../train_data_1/masked_images'
TRAIN_MASK_DIR = '../../train_data_1/masks'

JSON_DIR = TRAIN_JSON_DIR
MASK_DIR = TRAIN_MASK_DIR
IMAGE_DIR = TRAIN_IMG_DIR
IMAGE_OUT_DIR = TRAIN_MASKED_IMAGES_DIR
