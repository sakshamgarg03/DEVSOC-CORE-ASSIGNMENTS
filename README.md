# DEVSOC-CORE-ASSIGNMENTS

## PROJECT 1 :NEURAL STYLE TRANSFER USING VGG-19

### ABOUT
This project implements Neural Style Transfer (NST) using the pre-trained VGG-19 convolutional neural network from PyTorch.

The goal is to generate a new image that preserves the content of one image while adopting the artistic style of another. This is achieved by optimizing a randomly initialized image to minimize a combination of content loss and style loss:

Content loss ensures the output image retains the overall structure of the content image.

Style loss ensures the output image captures the textures, colors, and brushstrokes of the style image by comparing Gram matrices of feature maps.
This is an implementation of <a href = "https://drive.google.com/file/d/1Dbxaazv-L2SbC3gY4cPlqOQmM2iGmwyB/view">CNN-based image style transformation--Using VGG19 </a>.

### LIBRARIES
- PYTORCH
- MATPLOTLIB
### HOW TO RUN
The code is written using google colab notebooks.

+ Open the link : <a href = "https://colab.research.google.com/drive/167rGjsEr3fUt9iJbI5d9tuPdLMF8SceM?usp=sharing">neural_style_transfer.ipynb</a>

+ Select the GPU Option in the RUNTIME Menu.

+ #### Upload
  - content.jpg
  - style.jpg

+ Run all the cells to execute



## PROJECT 2 :NEURAL NETWORK USING MNIST DATASET

### ABOUT

This project implements a simple feedforward neural network from scratch (without using PyTorch or TensorFlow) to classify handwritten digits from the popular MNIST dataset.

It serves as an educational example to understand how neural networks work under the hood — covering every step from data loading and preprocessing to forward propagation, backpropagation, and training.
