import numpy as np
import cv2
import os
import tqdm
from torchvision import transforms as T
from constant import *
from PIL import Image

image_name_list = os.listdir(IMAGE_DIR)
for image in tqdm.tqdm(image_name_list):
    img = Image.open(os.path.join(IMAGE_DIR, image))
    color_aug = T.ColorJitter(brightness=0.4, contrast=0.4)
    img_aug = color_aug(img)
    image = os.path.join(AUGMENTATION_DIR, image)
    img_aug = np.array(img_aug)
    cv2.imwrite(image, img_aug)

mask_name_list = os.listdir(MASK_DIR)
for mask in tqdm.tqdm(mask_name_list):
    msk = cv2.imread(os.path.join(MASK_DIR, mask))
    mask = os.path.join(AUGMENTATION_MASK_DIR, mask)
    cv2.imwrite(mask, msk)