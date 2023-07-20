# Questions and Answers
---
## Contents
---
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
### - What is Machine Learning?

Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.

---

### - What is Unsupervised vs Supervised learning difference?

The main difference between supervised and unsupervised learning is the type of input data required. Supervised machine learning requires labeled training data, while unsupervised learning relies on unlabeled raw data.

---

### - What is Deep Learning?

Deep learning is an advanced type of machine learning that uses multilayered neural networks to establish nested hierarchical models for data processing and analysis, as in image recognition or natural language processing, with the goal of self-directed information processing.

---

### - What is Neural Network (NN)? 

Artificial Neural Networks (ANNs) are computer systems developed with the aim of automatically performing tasks such as deriving new information, generating new knowledge, and making discoveries through the process of learning, similar to the human brain's capabilities, without any external assistance.

Artificial Neural Networks are inspired by the human brain and are a result of mathematically modeling the learning process. They mimic the structure of biological neural networks in the brain and imitate their abilities in learning, memory, and generalization. In artificial neural networks, the learning process is achieved using examples. Input and output information is provided during the learning process, and rules are established.

---

### - What is Convolution Neural Network (CNN)? Please give 2 advantages over NN.

In deep learning, a convolutional neural network (CNN) is a class of artificial neural network most commonly applied to analyze visual imagery. CNNs use a mathematical operation called convolution in place of general matrix multiplication in at least one of their layers. They are specifically designed to process pixel data and are used in image recognition and processing.

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

### - What is data and dataset difference? 

**Data** are observations or measurements (unprocessed or processed) represented as text, numbers, or multimedia.

A **dataset** is a structured collection of data generally associated with a unique body of work.

### - What is the difference between supervised and unsupervised learning in terms of dataset?

In a supervised learning model, the algorithm learns on a labeled dataset, providing an answer key that the algorithm can use to evaluate its accuracy on training data. 

An unsupervised model, in contrast, provides unlabeled data that the algorithm tries to make sense of by extracting features and patterns on its own.

    Supervised   => Input and label

    Unsupervised => Input


