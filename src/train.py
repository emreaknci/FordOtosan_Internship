import shutil
import torch
import tqdm
from model2 import FoInternNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from constant import *
from predict import predict
######### PARAMETERS #########
valid_size = 0.3
test_size = 0.1
batch_size = 4
epochs = 20
cuda = False
input_shape = (224, 224)
n_classes = 2
###############################

# MODEL PATH
modelPath = os.path.join("../models/model2.pt")

# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

# DATA CHECK
image_mask_check(image_path_list, mask_path_list)

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]

# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size

# CALL MODEL

# model = FoInternNet(n_channels=3, n_classes=2, bilinear=True)
model = FoInternNet(n_classes=2)


# DEFINE LOSS FUNCTION AND OPTIMIZER
# criterion = nn.BCELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

criterion =  nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()

val_losses = []
train_losses = []

print(test_input_path_list.__len__())
print(valid_input_path_list.__len__())

# COPY TEST IMAGES TO PREDICT DIRECTORY
for item in test_input_path_list:
    shutil.copy(item, os.path.join(PREDICT_DIR, os.path.basename(item)))

# TRAINING THE NEURAL NETWORK
def train():
    for epoch in tqdm.tqdm(range(epochs)):
        running_loss = 0
        for ind in range(steps_per_epoch):
            batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(
                ind+1)]
            batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(
                ind+1)]
            batch_input = tensorize_image(
                batch_input_path_list, input_shape, cuda)
            batch_label = tensorize_mask(
                batch_label_path_list, input_shape, n_classes, cuda)

            optimizer.zero_grad()

            outputs = model(batch_input)


            loss = criterion(outputs, batch_label)
            print(loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, epochs, ind + 1, steps_per_epoch, loss.item()))
            if ind == steps_per_epoch-1:
                train_losses.append(running_loss)
                print('training loss on epoch {}: {}'.format(epoch, running_loss))
                val_loss = 0
                for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                    batch_input = tensorize_image(
                        [valid_input_path], input_shape, cuda)
                    batch_label = tensorize_mask(
                        [valid_label_path], input_shape, n_classes, cuda)
                    outputs = model(batch_input)
                    loss = criterion(outputs, batch_label)
                    val_loss += loss.item()
                    val_losses.append(val_loss)
                    break

                print('validation loss on epoch {}: {}'.format(epoch, val_loss))

    torch.save(model, modelPath)


def draw_graph(train_losses, val_losses):
    loss_train = [float(i)/sum(train_losses) for i in train_losses]
    loss_val = [float(i)/sum(val_losses) for i in val_losses]
    epochs = list(range(1, epochs+1, 1))
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


train()
draw_graph(train_losses, val_losses)
predict(glob.glob(os.path.join(PREDICT_DIR, '*')))
