import torch
from preprocess import tensorize_image
import os
import glob
import numpy as np
import tqdm
import cv2
from constant import *
from accuracy import accuracy
input_shape = (224, 224)
cuda = False
IMAGE_DIR = os.path.join(PREDICT_DIR)
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()


model = torch.load(r"../models/model1.pt", map_location=torch.device('cpu'))
model.eval()
print(type(model))
if cuda:
    model = model.cuda()


test_input_path_list = image_path_list


def predict(test_input_path_list):
    for i in tqdm.tqdm(range(len(test_input_path_list))):
        batch_test = test_input_path_list[i:i+1]
        test_input = tensorize_image(batch_test, input_shape, cuda)
        outs = model(test_input)
        out = torch.argmax(outs, axis=1)
        out_cpu = out.cpu()
        outputs_list = out_cpu.detach().numpy()
        mask = np.squeeze(outputs_list, axis=0)
        mask = np.squeeze(outputs_list, axis=0)
        mask = cv2.resize(mask.astype(np.uint8), (1920, 1208))

        mask = mask * 100
        predict_mask_path = os.path.join(PREDICT_MASK_DIR, batch_test[0])
        predict_mask_path = predict_mask_path.replace(
            'predicts', 'predict_masks')

        cv2.imwrite(predict_mask_path, mask.astype(np.uint8))


def write_accuracy_on_image(accuracies):
    for i in tqdm.tqdm(range(len(test_input_path_list))):
        batch_test = test_input_path_list[i:i+1]
        test_input = tensorize_image(batch_test, input_shape, cuda)
        outs = model(test_input)
        out = torch.argmax(outs, axis=1)
        out_cpu = out.cpu()
        outputs_list = out_cpu.detach().numpy()
        mask = np.squeeze(outputs_list, axis=0)

        img = cv2.imread(batch_test[0])
        mask = cv2.resize(mask.astype(np.uint8), (1920, 1208))
        cpy_img = img.copy()
        img[mask == 1, :] = (0, 255, 0)
        opac_image = (img/2+cpy_img/2).astype(np.uint8)
        predict_name = batch_test[0]
        predict_path = predict_name.replace(
            'predicts', 'MODEL_AUG1')
        font_color = (255, 255, 255)
        texts = [
            {
                'text': "First Model",
                'org': (50, 50),
                'font': cv2.FONT_HERSHEY_SIMPLEX,
                'font_scale': 1,
                'font_thickness': 2,
            },
            {
                'text': "Augmentation: True",
                'org': (50, 100),
                'font': cv2.FONT_HERSHEY_SIMPLEX,
                'font_scale': 1,
                'font_thickness': 2,
            },
            {
                'text': f"Accuracy: {accuracies[i]:.2f}%",
                'org': (50, 150),
                'font': cv2.FONT_HERSHEY_SIMPLEX,
                'font_scale': 1,
                'font_thickness': 2,
            },
        ]

        for text_info in texts:
            org = text_info['org']
            font = text_info['font']
            font_scale = text_info['font_scale']
            font_thickness = text_info['font_thickness']
            font_color = (255, 255, 255)
            text = text_info['text']

            cv2.putText(opac_image, text, org, font,
                        font_scale, font_color, font_thickness)
        cv2.imwrite(predict_path, opac_image)


if __name__ == "__main__":
    predict(test_input_path_list)
    accuracies = accuracy()
    write_accuracy_on_image(accuracies)
