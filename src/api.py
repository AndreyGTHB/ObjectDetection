import numpy as np
from keras import layers, optimizers, models


def relu(x):
    if x > 0:
        return x
    return 0


def conv(input_image, filters, kernel_size=(3, 3), padding=0, activation=relu):
    input_image = input_image.tolist()
    for i in range(1, padding+1):
        input_image.insert(0, np.zeros())

    width, height, _ = input_image.shape
    output = np.array([[np.zeros(filters) for h in height] for w in width])

    for w in range(width):
        for h in range(height):
            for f in range(filters):

