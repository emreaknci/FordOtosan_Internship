import cv2
import numpy as np
from constant import *


def accuracy():
    true_masked_images = MASK_DIR
    predict_masked_images = PREDICT_MASK_DIR

    true_masked_image_files = os.listdir(true_masked_images)
    predict_masked_image_files = os.listdir(predict_masked_images)

    # A list is created to store files with the same name
    matching_files = []

    # Her bir tahmin edilen dosya adını incele
    for predict_image_file in predict_masked_image_files:
        # If this filename exists in true_masked_images_files add to list
        if predict_image_file in true_masked_image_files:
            matching_files.append(predict_image_file)

    
    filtered_true_masked_image_files = [
        file for file in true_masked_image_files if file in matching_files]


    total_mismatch_percantage = 0
    i = 0
    print("Calculating accuracy....")
    for true_image_path, predict_image_path in zip(filtered_true_masked_image_files, predict_masked_image_files):
        i += 1

        true_path = os.path.join(true_masked_images, true_image_path)
        predict_path = os.path.join(predict_masked_images, predict_image_path)

        true_img = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)
        predict_img = cv2.imread(predict_path, cv2.IMREAD_GRAYSCALE)
        difference = cv2.absdiff(true_img, predict_img)

        # Farklı piksellerin sayısını hesapla
        non_matching_pixels = np.count_nonzero(difference)

        total_pixels = true_img.size

        mismatch_percentage = (non_matching_pixels / total_pixels) * 100
        total_mismatch_percantage += mismatch_percentage

    print(f"Accuracy: %{100-total_mismatch_percantage/i}")
