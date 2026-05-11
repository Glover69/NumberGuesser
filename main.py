import os
import random
from typing import Any

import matplotlib.pyplot as plt

from data.data_loader import MnistDataloader


training_images_filepath = './data/train-images.idx3-ubyte'
training_labels_filepath = './data/train-labels.idx1-ubyte'
test_images_filepath = './data/val/t10k-images.idx3-ubyte'
test_labels_filepath = './data/val/t10k-labels.idx1-ubyte'

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize = 15);
            print(title_text)
        index += 1

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


# converting the images from (0 - 255) to (0 - 1) for our activations in the N.N
train_nn = []
test_nn = []

for x in range(len(x_train)):
    x_train[x] = x_train[x]/255
    train_nn.append(x_train[x])

for x in range(len(x_test)):
    x_test[x] = x_test[x]/255
    test_nn.append(x_test[x])