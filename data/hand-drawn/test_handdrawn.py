# import numpy as np
# from PIL import Image
#
# import network
#
# # Open the image file
# img = Image.open("1.png").convert('L')
# img.save('1-grayscale.png')
#
# # image is already 28 by 28, but we'll resize again
# resized_img = img.resize((28, 28))
#
# # normalizes image to 0 to 1
# img_guess = np.array(img) / 255.0
#
# Image.fromarray((img_guess * 255).astype(np.uint8)).save('debug.png')
#
#
# network = network.Network()
#
# prediction = network.forward(img_guess)
#
# print(np.argmax(prediction))
#
#
# # img.show()