import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_mnist_from_csv
from layer import Layer
from activations import sigmoid, sigmoid_derivative, relu, relu_derivative,softmax,cross_entropy


train_data = load_mnist_from_csv("data/train.csv", limit=5000)
test_data = load_mnist_from_csv("data/test.csv", limit=400)

layer1 = Layer(784, 64, relu, relu_derivative)
layer2 = Layer(64, 10, softmax, None)

# Training Parameters (using different para changes accuracy)
learning_rate = 0.01
epochs = 20


train_losses = []
train_accuracies = []
test_accuracies = []

# Training Loop
for epoch in range(epochs):
    total_loss = 0
    correct = 0

    for x, y in train_data:
        # Forward
        a1 = layer1.forward(x)
        a2 = layer2.forward(a1)

        # Loss
        loss = cross_entropy(a2,y)
        total_loss += loss

        # Accuracy
        if np.argmax(a2) == np.argmax(y):
            correct += 1
        lambda_=0.0005
        # Backward
        d_output = (a2 - y)
        d_hidden = layer2.backward(d_output, learning_rate,lambda_)
        layer1.backward(d_hidden, learning_rate,lambda_)

    # Evaluate on test set
    test_correct = 0
    for x, y in test_data:
        a1 = layer1.forward(x)
        a2 = layer2.forward(a1)
        if np.argmax(a2) == np.argmax(y):
            test_correct += 1


    avg_loss = total_loss / len(train_data)
    train_acc = correct / len(train_data)
    test_acc = test_correct / len(test_data)

    # Store for plotting
    train_losses.append(avg_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1} | Train Acc: {train_acc*100:.8f}% | "
          f"Test Acc: {test_acc*100:.8f}% | Loss: {avg_loss:.4f}")

# Plotting
epochs_range = range(1, epochs + 1)
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, marker='o', label="Train Loss", color='blue')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, marker='o', label="Train Accuracy", color='green')
plt.plot(epochs_range, test_accuracies, marker='x', label="Test Accuracy", color='red')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
