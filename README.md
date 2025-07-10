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
---
### HOW TO RUN
The code is written using google colab notebooks.

+ Open the link : <a href = "https://colab.research.google.com/drive/167rGjsEr3fUt9iJbI5d9tuPdLMF8SceM?usp=sharing">neural_style_transfer.ipynb</a>

+ Select the GPU Option in the RUNTIME Menu.
  
+ #### Upload
  - content.jpg
  - style.jpg

+ Run all the cells to execute

---

## PROJECT 2 :NEURAL NETWORK USING MNIST DATASET

### ABOUT

This project implements a simple feedforward neural network from scratch (without using PyTorch or TensorFlow) to classify handwritten digits from the popular MNIST dataset.

It serves as an educational example to understand how neural networks work under the hood — covering every step from data loading and preprocessing to forward propagation, backpropagation, and training.


<a href = "https://drive.google.com/file/d/1gZVSu9wGjcYM7_pejKz4XoVDIOQEJlBY/view?usp=drive_link" >train.csv</a>.
This is the link for the data set
(use the same for the test and train after naming one each as test dataset lacks the required labels)
---
###  Neurons & Layers

- The network is organized into layers, each containing **neurons**.
- Each layer has:
  - A **weight matrix** and **bias vector**.
  - An **activation function** (e.g., ReLU, Softmax).
- A `Layer` class handles all operations for one layer:
  - Forward propagation through the layer.
  - Backward propagation to update weights and biases.

---

###  Forward Propagation

- The input image is passed through the network layer by layer.
- At each layer, we compute:
  \[
  z = W \cdot x + b
  \]
  \[
  a = \text{activation}(z)
  \]
- The final layer uses **Softmax** to produce class probabilities.
- The output is compared to the true label using **Cross-Entropy Loss**.

---

###  Backpropagation

- The loss is propagated **backward** through the network.
- Gradients of the loss w.r.t. weights and biases are computed using the **chain rule**.
- For each layer:
  - Compute the gradient of the activation and then of \( z \).
  - Update weights and biases using **Stochastic Gradient Descent (SGD)**:
    \[
    W = W - \eta \cdot \frac{\partial L}{\partial W}
    \]
- This step allows the network to learn patterns from the data.

---

### Optimization & Training

- **Optimizer**: SGD (Stochastic Gradient Descent)
- **Loss Function**: Cross-Entropy
- **Activation Functions**:
  - ReLU for hidden layers
  - Softmax for output layer
 
---

### HOW TO RUN
- Make sure to have all the requirements
  - Python 3.X
  - NumPy
  - Matplotlib
- Make sure to have all the files as in the MNIST_NEURAL_NETWORK
  - activation.py
  - main.py
  - data_loader.py
  - layer.py
  - data/
      - train.csv
      - test.csv
        
---
## MADE BY : SAKSHAM GARG
  
