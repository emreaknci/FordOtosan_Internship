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
---
### Data Preprocessing
**Extracting Masks**
- [What is color space?](#--what-is-color-space)
- [What RGB stands for?](#--what-rgb-stands-for)
- [In Python, can we transform from one color space to another?](#--in-python-can-we-transform-from-one-color-space-to-another)
- [What is the popular library for image processing?](#--what-is-the-popular-library-for-image-processing)
---
### Converting into Tensor
- [What is Computational Graph](#--what-is-computational-graph)
- [What is Tensor?](#--what-is-tensor)
- [What is one hot encoding](#--what-is-one-hot-encoding)
- [What is CUDA programming?](#--what-is-cuda-programming)

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

***Below are two advantages of CNNs over regular Neural Networks:***

1. **Feature Learning:** CNNs automatically learn complex features from input data. Instead of using a regular input layer, CNNs employ a specific convolutional layer to extract basic features from the data, reducing the need for human intervention.
2. **Parameter Sharing:** CNNs are more efficient in their learning process by sharing weights. Instead of having separate weights for each feature, weights are shared across different positions, resulting in fewer parameters in the model.

---

### - What is segmentation task in NN? Is it supervised or unsupervised?

Segmentation task in NN is a task used to group or separate objects or regions in an input image based on specific features or attributes. Image segmentation involves the process of determining and isolating the boundaries or regions of objects.

***The segmentation task can be performed in both supervised and unsupervised ways.*** 

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





