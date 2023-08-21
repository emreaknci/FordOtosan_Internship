import os

# Path to train data

TRAIN_IMG_DIR = '../../train_data/images'
TRAIN_JSON_DIR = '../../train_data/jsons'
TRAIN_MASKED_IMAGES_DIR = '../../train_data/masked_images'
TRAIN_MASK_DIR = '../../train_data/masks'

# Path to jsons
JSON_DIR = '../data/jsons'
if not os.path.exists(JSON_DIR):
    os.mkdir(JSON_DIR)
    
# Path to mask
MASK_DIR  = '../data/masks'
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)

# Path to output images
IMAGE_OUT_DIR = '../data/masked_images'
if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(IMAGE_OUT_DIR)

# Path to original images
IMAGE_DIR = '../data/images'


# In order to visualize masked-image(s), change "False" with "True"
VISUALIZE = False

# Bacth size
BATCH_SIZE = 4

# Input dimension
HEIGHT = 224
WIDTH = 224

# Number of class, for this task it is 2: Non-drivable area and Driviable area
N_CLASS= 2

JSON_DIR=TRAIN_JSON_DIR
MASK_DIR=TRAIN_MASK_DIR
IMAGE_DIR=TRAIN_IMG_DIR
IMAGE_OUT_DIR=TRAIN_MASKED_IMAGES_DIR