from pathlib import Path

import numpy as np
from PIL import Image

import network
from data.data_loader import MnistDataloader, DataLoader
from test import process_handdrawn

training_images_filepath = './data/train-images.idx3-ubyte'
training_labels_filepath = './data/train-labels.idx1-ubyte'
test_images_filepath = './data/val/t10k-images.idx3-ubyte'
test_labels_filepath = './data/val/t10k-labels.idx1-ubyte'

#
# Load MINST dataset
#
# mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
# (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
data_loader = DataLoader()
(x_train, y_train), (x_test, y_test) = data_loader.load_data()


# converting the images from (0 - 255) to (0 - 1) for our activations in the N.N
x_train = np.array(x_train) / 255.0
x_test = np.array(x_test) / 255.0
y_train = np.array(y_train)
y_test = np.array(y_test)



# now we run the training loop
network = network.Network()

print(len(x_train))

train_steps = 0
epoch_num = 5

for e, epoch in enumerate(range(epoch_num)):
    print(f"-------EPOCH {e + 1}--------")
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    for index, img in enumerate(x_train):
        # we get the first prediction in
        prediction = network.forward(img)

        # we then get our loss w the cost function
        loss = network.get_loss(y_train[index])

        # Then we run our backprop logic, update the weights and move on to the next img
        network.backward_prop(img)

        if train_steps % 1000 == 0:
            correct = 0
            for index2, i in enumerate(x_test):
                prediction = network.forward(i)
                if np.argmax(prediction) == y_test[index2]:
                    correct += 1

            print(f"print accuracy: {correct}/{len(x_test)}")
        train_steps += 1


test_dir = "/Users/danielglover/Python/Forge/output-test/val"
for file in Path(test_dir).glob("*.jpg"):
    pixels = process_handdrawn(file)

    resized_check = Image.fromarray(pixels)
    resized_check.save("debug_output.png")

    pixels = pixels / 255.0  # match training normalization

    prediction = network.forward(pixels)
    print(file.name, "→ predicted:", np.argmax(prediction))


# def test_handdrawn(folder):
#     for file in Path(folder).glob("*.jpg"):
#         pixels = process_handdrawn(file)
#         pixels = pixels / 255.0          # match training normalization
#         prediction = network.forward(pixels)
#         print(file.name, "→ predicted:", np.argmax(prediction))
#
# test_handdrawn("./data/hand-drawn")


# Open the image file
# img = Image.open("./data/hand-drawn/1.png").convert('L')
# img.save('./data/hand-drawn/1-grayscale.png')
#
# # image is already 28 by 28, but we'll resize again
# resized_img = img.resize((28, 28))
#
# # normalizes image to 0 to 1
# img_guess = np.array(img) / 255.0
#
# print(img_guess.shape)
# print(img_guess.max(), img_guess.min())
#
# prediction = network.forward(img_guess)
#
# print(f'Prediction for hand-drawn img: {np.argmax(prediction)}')


# img.show()