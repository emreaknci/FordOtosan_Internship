"""
We need to prepare the data to feed the network: we have - data/masks, data/images - directories where we prepared masks and input images. Then, convert each file/image into a tensor for our purpose.

We need to write two functions in src/preprocess.py:
    - one for feature/input          -> tensorize_image()
    - the other for mask/label    -> tensorize_mask()


Our model will accepts the input/feature tensor whose dimension is
[batch_size, output_shape[0], output_shape[1], 3]
&
the label tensor whose dimension is
[batch_size, output_shape[0], output_shape[1], 2].

At the end of the task, our data will be ready to train the model designed.

"""


import glob
import cv2
import torch
import numpy as np
from constant import *


def tensorize_image(image_path_list, output_shape, cuda=False):
    """


    Parameters
    ----------
    image_path_list : list of strings
        [“data/images/img1.png”, .., “data/images/imgn.png”] corresponds to
        n images to be trained each step.
    output_shape : tuple of integers
        (n1, n2): n1, n2 is width and height of the DNN model’s input.
    cuda : boolean, optional
        For multiprocessing,switch to True. The default is False.

    Returns
    -------
    torch_image : Torch tensor
        Batch tensor whose size is [batch_size, output_shape[0], output_shape[1], C].       For this case C = 3.

    """
    # Create empty list
    local_image_list = []

    # For each image
    for image_path in image_path_list:

        print(image_path)
        # Access and read image
        image = cv2.imread(image_path)

        # Resize the image according to defined shape
        image = cv2.resize(image, output_shape)

        # Change input structure according to pytorch input structure
        torchlike_image = torchlike_data(image)

        # Add into the list
        local_image_list.append(torchlike_image)

    # Convert from list structure to torch tensor

    image_array = np.array(local_image_list, dtype=np.float32)
    torch_image = torch.from_numpy(image_array).float()

    # If multiprocessing is chosen
    if cuda:
        torch_image = torch_image.cuda()

    return torch_image

def tensorize_mask(mask_path_list, output_shape, n_class, cuda=False):
    """


    Parameters
    ----------
    mask_path_list : list of strings
        [“data/masks/mask1.png”, .., “data/masks/maskn.png”] corresponds
        to n masks to be used as labels for each step.
    output_shape : tuple of integers
        (n1, n2): n1, n2 is width and height of the DNN model’s input.
    n_class : integer
        Number of classes.
    cuda : boolean, optional
        For multiprocessing, switch to True. The default is False.

    Returns
    -------
    torch_mask : TYPE
        DESCRIPTION.

    """

    # Create empty list
    local_mask_list = []

    # For each masks
    for mask_path in mask_path_list:

        # Access and read mask
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            print(f"Could not read mask from: {mask_path}")
            continue  # Skip this mask and move on to the next one
        # Resize the image according to defined shape
        mask = cv2.resize(mask, output_shape, interpolation = cv2.INTER_NEAREST)

        # Apply One-Hot Encoding to image
        mask = one_hot_encoder(mask, n_class)

        # Change input structure according to pytorch input structure
        torchlike_mask = torchlike_data(mask)


        local_mask_list.append(torchlike_mask)

    mask_array = np.array(local_mask_list, dtype=np.int64)
    torch_mask = torch.from_numpy(mask_array).float()
    if cuda:
        torch_mask = torch_mask.cuda()

    return torch_mask

def image_mask_check(image_path_list, mask_path_list):
    
    if len(image_path_list) != len(mask_path_list):
        print("There are probably missing files. The number of input images and mask images is not the same.")
        return False

    
    for img_path, mask_path in zip(image_path_list, mask_path_list):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_name = os.path.splitext(os.path.basename(mask_path))[0]

        if img_name != mask_name:
            print(f"Image and mask names do not match for {img_path} and {mask_path}.")
            return False

    return True

############################ TODO ################################
def torchlike_data(data):
    n_channels = data.shape[2]
    torchlike_data = np.empty((n_channels, data.shape[0], data.shape[1]))#Returns a new array of the given shape and type.
    #creates an array of these sizes
    for ch in range(n_channels):# generates ch numbers as long as the list
        torchlike_data[ch] = data[:,:,ch] #torchlike_data[0]=data[:,:,0] 
        #Export data in data individually to torchlike_data
    return torchlike_data

def one_hot_encoder(data, n_class):
    global one_hot
    # one hot encode
    # Create an np.array of zeros.
    one_hot = np.zeros((data.shape[0], data.shape[1], n_class), dtype=np.int32)  # Updated dtype
    # Find unique values in res_mask [0,1]
    # increase in i by the length of the list
    # [0,1] when returning the inside of list, each list element is given to unique_value variable
    global unique_values
    unique_values = np.unique(data)

    for i, unique_value in enumerate(np.unique(data)):
        one_hot[:, :, i][data == unique_value] = 1
    return one_hot
############################ TODO END ################################





if __name__ == '__main__':

    # Access images
    image_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
    image_list.sort()


    # Access masks
    mask_list = glob.glob(os.path.join(MASK_DIR, '*'))
    mask_list.sort()


    # Check image-mask match
    if image_mask_check(image_list, mask_list):
        # Take image to number of batch size
        batch_image_list = image_list[:BATCH_SIZE]
        global batch_image_tensor
        # Convert into torch tensor
        batch_image_tensor = tensorize_image(batch_image_list, (HEIGHT,WIDTH))
        
        # Check
        print("For features:\ndtype is " + str(batch_image_tensor.dtype))
        print("Type is " + str(type(batch_image_tensor)))
        print("The size should be [" + str(BATCH_SIZE) + ", 3, " + str(HEIGHT) + ", " + str(WIDTH) + "]")
        print("Size is " + str(batch_image_tensor.shape)+"\n")
        
        # Take masks to number of batch size
        batch_mask_list = mask_list[:BATCH_SIZE]
        print("--------------------------------------------------")
        # Convert into torch tensor
        batch_mask_tensor = tensorize_mask(batch_mask_list, (HEIGHT,WIDTH), 2)
        
        # Check
        print("For labels:\ndtype is "+str(batch_mask_tensor.dtype))
        print("Type is "+str(type(batch_mask_tensor)))
        print("The size should be ["+str(BATCH_SIZE)+", 2, "+str(HEIGHT)+", "+str(WIDTH)+"]")
        print("Size is "+str(batch_mask_tensor.shape))