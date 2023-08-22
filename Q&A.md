# Questions and Answers

## Contents

### Fundamentals

- [What is Machine Learning?](#--what-is-machine-learning)
- [What is Unsupervised vs Supervised learning difference?](#--what-is-unsupervised-vs-supervised-learning-difference)
- [What is Deep Learning?](#--what-is-deep-learning)
- [What is Neural Network (NN)?](#--what-is-neural-network-nn)
- [What is Convolution Neural Network (CNN)? Please give 2 advantages over NN.](#--what-is-convolution-neural-network-cnn-please-give-2-advantages-over-nn)
- [What is segmentation task in NN? Is it supervised or unsupervised?](#--what-is-segmentation-task-in-nn-is-it-supervised-or-unsupervised)
- [What is classification task in NN? Is it supervised or unsupervised?](#--what-is-classification-task-in-nn-is-it-supervised-or-unsupervised)
- [Compare segmentation and classification in NN.](#--compare-segmentation-and-classification-in-nn)
- [What is data and dataset difference?](#--what-is-data-and-dataset-difference)
- [What is the difference between supervised and unsupervised learning in terms of dataset?](#--what-is-the-difference-between-supervised-and-unsupervised-learning-in-terms-of-dataset)

### Data Preprocessing

**Extracting Masks**

- [What is color space?](#--what-is-color-space)
- [What RGB stands for?](#--what-rgb-stands-for)
- [In Python, can we transform from one color space to another?](#--in-python-can-we-transform-from-one-color-space-to-another)
- [What is the popular library for image processing?](#--what-is-the-popular-library-for-image-processing)

### Converting into Tensor

- [What is Computational Graph](#--what-is-computational-graph)
- [What is Tensor?](#--what-is-tensor)
- [What is one hot encoding](#--what-is-one-hot-encoding)
- [What is CUDA programming?](#--what-is-cuda-programming)

### Design Segmentation Model

- [What is the difference between CNN and Fully CNN (FCNN)?](#--what-is-the-difference-between-cnn-and-fully-cnn-fcnn-)
- [What are the different layers on CNN?](#--what-are-the-different-layers-on-cnn)
- [What is activation function ? Why is softmax usually used in the last layer?](#--what-is-activation-function--why-is-softmax-usually-used-in-the-last-layer)

### Train

- [What is parameter and hyper-parameter in NN ?](#--what-is-parameter-and-hyper-parameter-in-nn-)
- [What is the validation dataset ?](#--what-is-the-validation-dataset)
- [What is an epoch ?](#--what-is-an-epoch)
- [What is batch ?](#--what-is-batch)
- [What is iteration ?](#--what-is-iteration)
- [What is Cost Function ?](#--what-is-the-cost-function)
- [What is/are the purpose(s) of an optimizer in NN ?](#--what-isare-the-purposes-of-an-optimizer-in-nn)
- [What is Batch Gradient Descent & Stochastic Gradient Descent ?](#--what-is-batch-gradient-descent--stochastic-gradient-descent)
- [What is Backpropogation ? What is used for ?](#--what-is-backpropogation--what-is-used-for-)

---

## Fundamentals

### - What is Machine Learning?

Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.

---

### - What is Unsupervised vs Supervised learning difference?

The main difference between supervised and unsupervised learning is the type of input data required. Supervised machine learning requires labeled training data, while unsupervised learning relies on unlabeled raw data.

<p align="center">
    <img src="/images/difference-between-supervised-and-unsupervised-learning.jpg"/>
</p>

---

### - What is Deep Learning?

Deep learning is an advanced type of machine learning that uses multilayered neural networks to establish nested hierarchical models for data processing and analysis, as in image recognition or natural language processing, with the goal of self-directed information processing.

---

### - What is Neural Network (NN)?

Artificial Neural Networks (ANNs) are computer systems developed with the aim of automatically performing tasks such as deriving new information, generating new knowledge, and making discoveries through the process of learning, similar to the human brain's capabilities, without any external assistance.

Artificial Neural Networks are inspired by the human brain and are a result of mathematically modeling the learning process. They mimic the structure of biological neural networks in the brain and imitate their abilities in learning, memory, and generalization. In artificial neural networks, the learning process is achieved using examples. Input and output information is provided during the learning process, and rules are established.

<p align="center">
    <img src="/images/220px-Neural_network_example.svg.png"/>
</p>

---

### - What is Convolution Neural Network (CNN)? Please give 2 advantages over NN.

In deep learning, a convolutional neural network (CNN) is a class of artificial neural network most commonly applied to analyze visual imagery. CNNs use a mathematical operation called convolution in place of general matrix multiplication in at least one of their layers. They are specifically designed to process pixel data and are used in image recognition and processing.

<p align="center">
    <img src="/images/CNN_recognizes_a_bird1.webp"/>
</p>

**_Below are two advantages of CNNs over regular Neural Networks:_**

1. **Feature Learning:** CNNs automatically learn complex features from input data. Instead of using a regular input layer, CNNs employ a specific convolutional layer to extract basic features from the data, reducing the need for human intervention.
2. **Parameter Sharing:** CNNs are more efficient in their learning process by sharing weights. Instead of having separate weights for each feature, weights are shared across different positions, resulting in fewer parameters in the model.

---

### - What is segmentation task in NN? Is it supervised or unsupervised?

Segmentation task in NN is a task used to group or separate objects or regions in an input image based on specific features or attributes. Image segmentation involves the process of determining and isolating the boundaries or regions of objects.

**_The segmentation task can be performed in both supervised and unsupervised ways._**

In supervised segmentation, the network is provided with target (label) masks corresponding to the input images in the training data. These masks are pixel-wise labels that indicate the boundaries and regions of objects. The network is trained with these annotated data and predicts the object boundaries or regions in new images.

In unsupervised segmentation, there is no labeling in the training data, and the network automatically segments the images by learning structures or regions with similar features in the data.

---

### - What is classification task in NN? Is it supervised or unsupervised?

A classification task in neural networks involves assigning given data to specific categories or classes using an artificial neural network or deep learning model. For example, it can be used to recognize objects in an image, separate emails into spam and non-spam categories, or diagnose diseases, among various other applications.

The classification task is typically considered a part of supervised learning.

---

### - Compare segmentation and classification in NN.

Classification refers to a type of labeling where an image/video is assigned certain concepts, with the goal of answering the question, “What is in this image/video?”

The classification process is easier than segmentation, in classification all objects in a single image is grouped or categorized into a single class. While in segmentation each object of a single class in an image is highlighted with different shades to make them recognizable to computer vision.

Segmentation is a type of labeling where each pixel in an image is labeled with given concepts. Here, whole images are divided into pixel groupings that can then be labeled and classified to simplify an image or change how an image is presented to the model to make it easier to analyze.

Segmentation models provide the exact outline of the object within an image. That is, pixel by pixel details are provided for a given object, as opposed to Classification models, where the model identifies what is in an image, and Detection models, which places a bounding box around specific objects.

---

### - What is data and dataset difference?

**Data** are observations or measurements (unprocessed or processed) represented as text, numbers, or multimedia.

A **dataset** is a structured collection of data generally associated with a unique body of work.

### - What is the difference between supervised and unsupervised learning in terms of dataset?

In a supervised learning model, the algorithm learns on a labeled dataset, providing an answer key that the algorithm can use to evaluate its accuracy on training data.

An unsupervised model, in contrast, provides unlabeled data that the algorithm tries to make sense of by extracting features and patterns on its own.

    Supervised   => Input and label

    Unsupervised => Input

---

## Data Preprocessing

### Extracting Masks

#### - What is color space?

**Color space** is a mathematical model used to represent colors. Different color spaces allow for the quantitative representation and processing of colors. Color spaces can include various color models and coordinate systems such as RGB (Red Green Blue), CMYK (Cyan Magenta Yellow Key), HSL (Hue Saturation Lightness), and more.

<p align="center">
    <img src="/images/Color-Space-Visualization-•-CIE-1931-Color-Space-.jpg" style="width:50%"/>
</p>

---

#### - What RGB stands for?

RGB means Red Green Blue, ie the primary colors in additive color synthesis.

A RGB file consists of composite layers of Red, Gree and Blue, each being coded on 256 levels from 0 to 255. For example, black corresponds to the levels R=0, G=0, B=0, and white corresponds to the levels R=255, G=255, B=255.

<p align="center">
    <img src="/images/lTkZrgPSO2ru8F6aIaQJ_rgb-color-mode.jpg"/>
</p>

---

#### - In Python, can we transform from one color space to another?

Yes, in Python, it is possible to transform from one color space to another. For this purpose, libraries like OpenCV and some others provide functions to perform color space transformations between different color spaces. For example, in OpenCV, you can use the `cv2.cvtColor()` function to perform color space transformation for an image.

---

#### - What is the popular library for image processing?

The most popular library for image processing is OpenCV. OpenCV is an open-source and powerful image processing library. It can be used to perform both simple image manipulations and develop more complex computer vision applications.

---

## Converting into Tensor

### - What is Computational Graph

A computational graph is defined as a directed graph where the nodes correspond to mathematical operations. Computational graphs are a way of expressing and evaluating a mathematical expression.
For example, here is a simple mathematical equation:

    p = x + y

We can draw a computational graph of the above equation as follows.

<p align="center">
    <img src="/images/computational_graph_equation1.jpg"/>
</p>

---

### - What is Tensor?

In mathematics, a tensor is an algebraic object that describes a multilinear relationship between sets of algebraic objects related to a vector space. Tensors may map between different objects such as vectors, scalars, and even other tensors. There are many types of tensors, including scalars and vectors (which are the simplest tensors), dual vectors, multilinear maps between vector spaces, and even some operations such as the dot product. Tensors are defined independent of any basis, although they are often referred to by their components in a basis related to a particular coordinate system; those components form an array, which can be thought of as a high-dimensional matrix.

<p align="center">
    <img src="/images/300px-Components_stress_tensor.svg.png"/>
</p>

---

### - What is One Hot Encoding?

One Hot Encoding refers to representing categorical variables as binary vectors. This process first requires mapping categorical values to integer values. Then, each integer value is represented as a binary vector with all values set to zero except for the index marked as 1. For example, consider the following data with 3 categories: apple, chicken, and broccoli. When these categories are binary encoded, the first row corresponds to apple with a 1, and the others are 0. The same process continues for the rest of the data, converting them into numerical representations.

<p align="center">
    <img src="/images/onehotencoding.jpg"/>
</p>

---

### - What is CUDA programming?

CUDA programming is a parallel computing platform and programming model developed by NVIDIA that enables developers to use GPUs (Graphics Processing Units) for general-purpose computing tasks.

---

## Design Segmentation Model

### - What is the difference between CNN and Fully CNN (FCNN) ?

CNNs were originally designed for tasks like image recognition. They usually take images as input and perform feature extraction using various convolutional layers, activation functions, and pooling layers. However, a typical CNN architecture reduces the input size, and the output often contains a single classification result. Therefore, CNNs are commonly used for classification tasks.

FCNNs, on the other hand, are designed to work at the pixel level. FCNN architectures consist entirely of convolutional layers, and the output at the final layers is usually kept at the same size as the input image. As a result, FCNNs are often used in pixel-level prediction tasks such as image segmentation, where the output can be a map indicating which class each pixel of the input image belongs to.

---

### - What are the different layers on CNN?

- Input layer
- Convo layer (Convo + ReLU)
- Pooling layer
- Fully connected(FC) layer
- Softmax/logistic layer
- Output layer

<p align="center">
    <img src="/images/cnn_layer.png"/>
</p>

1. Input Layer

   - Description: This is where the input data, typically an image, is introduced into the network. Each neuron in this layer represents a pixel value.
   - Purpose It acts as the starting point for the neural network, receiving the raw data to be processed.

2. Convolutional Layer:

   - Description: This layer applies a set of learnable filters (kernels) to the input data, extracting specific features by performing convolution operations, followed by an activation function such as Rectified Linear Unit (ReLU).
   - Purpose : It captures important patterns and features in the input data, such as edges, textures, and more complex structures.

3. Pooling Layer:

   - Description: This layer performs down-sampling by selecting the maximum or average value from a local region of the previous layer. It helps reduce the spatial dimensions of the data and makes the network more robust to variations in the input.
   - Purpose: It reduces the computational load and helps in maintaining important information while discarding less relevant details.

4. Fully Connected Layer:

   - Description: This layer connects each neuron from the previous layer to every neuron in the current layer, just like a traditional neural network. It learns complex combinations of features from the lower layers.
   - Purpose: It captures high-level features by considering interactions between the features learned in the convolutional layers, leading to better representation of the data.

5. Softmax/Logistic Layer (Softmax/Lojistik Katmanı):

   - Description: This layer is commonly used in classification tasks. It takes the output from the previous layer and converts it into a probability distribution over the different classes.
   - Purpose: It allows the network to predict the probability of each class, enabling the selection of the most likely class as the final prediction.

6. Output Layer (Çıkış Katmanı):

   - Description: This is the final layer of the network, producing the final output based on the problem being solved. For classification, it represents the predicted class label.
   - Purpose: It provides the ultimate result of the neural network's processing, whether it's a classification decision, regression prediction, or other task-specific output.

These layers can vary in number and order based on the specific CNN architecture used for a given task.

---

### - What is activation function ? Why is softmax usually used in the last layer?

An activation function is a mathematical function used in neural networks to determine the output of each neuron. This function takes the total input (weighted sum of inputs) of a neuron and transforms this output into a specific range or format. Activation functions are used in each layer of a neural network and can increase the complexity of the model, capture non-linear relationships, or emphasize specific features.

Softmax is an activation function commonly used in the last layer, especially in classification problems. The softmax function helps to transform the outputs into a probability distribution, aiding in predicting probabilities among different classes. It is particularly useful in multi-class classification problems (problems with multiple classes). Softmax normalizes the probabilities for each class so that they sum to 1, enabling the model to make more reliable predictions about which class an input belongs to.

---

## Train

### - What is parameter and hyper-parameter in NN ?

Parameters are the values learned by the neural network during the training process. They directly influence the network's ability to make predictions. In a typical neural network, parameters include weights and biases associated with each neuron and each layer. These values are updated during training using optimization algorithms like gradient descent, in order to minimize the difference between the predicted outputs and the actual target outputs.

Hyperparameters are settings that are not learned by the neural network itself but are set before the training process begins. They control various aspects of the training process and the architecture of the neural network. Examples of hyperparameters include learning rate, batch size, number of hidden layers, number of neurons in each layer, activation functions, and regularization strength. Tuning hyperparameters is essential to achieve optimal performance, and it often requires experimentation and testing.

- Number of hidden layers
- Learning rate
- Momentum
- Activation function
- Minibatch size
- Epochs
- Dropout rate

---

### - What is the validation dataset ?

A validation dataset is a portion of the data that is set aside during the training of a machine learning model. It is not used to train the model but is used to evaluate its performance and tune hyperparameters. After each training iteration (epoch), the model's performance is assessed on the validation dataset, helping to prevent overfitting and ensuring the model generalizes well to unseen data.

---

### - What is an epoch ?

An epoch in machine learning refers to a complete pass or iteration through the entire training dataset during the training of a model. In other words, one epoch means the model has seen and learned from every example in the dataset once. Multiple epochs are often used to fine-tune the model's parameters and improve its performance.

---

### - What is batch ?

In machine learning, a batch refers to a subset of the training data that is used together in a single iteration of the training process. Instead of updating the model's parameters after each individual example, batches allow multiple examples to be processed simultaneously, which can improve training efficiency and make better use of hardware resources.

---

### - What is iteration ?

In neural networks, an iteration typically refers to one complete cycle of forward and backward passes through a batch of training data. During an iteration, the model makes predictions on the batch, computes the loss, and then updates its parameters through backpropagation and optimization algorithms like gradient descent. Iterations are repeated multiple times (usually over many epochs) to train the model gradually.

---

### - What is Cost Function ?

It is a function that measures the performance of a Machine Learning model for given data. Cost Function quantifies the error between predicted values. A cost function measures “how good” a neural network did concerning its given training sample and the expected output. It also may depend on variables such as weights and biases.

---

### - What is/are the purpose(s) of an optimizer in NN ?

The purpose of an optimizer in a neural network is to adjust the model's parameters (such as weights and biases) during training in order to minimize the cost function. In other words, it's responsible for finding the optimal set of parameters that allows the neural network to make accurate predictions. Optimizers achieve this by iteratively updating the parameters based on the gradients of the cost function with respect to those parameters.

Common optimizers used in neural networks include Stochastic Gradient Descent (SGD), Adam, RMSprop, and more.

---

### - What is Batch Gradient Descent & Stochastic Gradient Descent ?
Batch Gradient Descent is an optimization algorithm used in machine learning to train neural networks and other models. In BGD, during each iteration of the training process, the entire training dataset is used to compute the gradient of the cost function with respect to the model parameters. This means that all the training examples are processed together in one large batch. Then, the model parameters are updated based on this computed gradient. BGD is known for its stable and deterministic convergence because it uses the complete dataset in each iteration, but it can be computationally expensive, especially for large datasets.

Stochastic Gradient Descent is another optimization algorithm used in machine learning. Unlike BGD, in SGD, only a single randomly selected training example (or a small random batch of examples) is used in each iteration to compute the gradient of the cost function. This randomness introduces noise into the optimization process, but it also makes SGD computationally more efficient, especially for large datasets. The noise can help the optimization process escape local minima. While SGD converges faster in some cases, it can have more oscillations in the optimization path compared to BGD.


---

### - What is Backpropogation ? What is used for ? 

Artificial neural networks use backpropagation as a learning algorithm to compute a gradient descent for weights. Desired outputs are compared to achieved system outputs, and then the systems are tuned by adjusting connection weights to narrow the difference between the two as much as possible. The algorithm gets its name because the weights are updated backward, from output towards input. A neural network propagates the signal of the input data forward through its parameters towards the moment of decision and then backpropagates information about the error, in reverse through the network, so that it can alter the parameters. This happens step by step:

1. The network guesses data, using its parameters
2. The network is measured with a loss function
3. The error is backpropagated to adjust the wrong-headed parameters

Backpropagation takes the error associated with a wrong guess by a neural network and uses that error to adjust the neural network’s parameters in the direction of less error. 
