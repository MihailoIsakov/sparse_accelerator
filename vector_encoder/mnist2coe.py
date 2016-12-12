import numpy as np
from keras.datasets import mnist
import coe_gen


def pad_to_length(data, length):
    pad_len = length - len(data)
    return np.pad(data, (0, pad_len), 'constant')


def get_mnist(pad=True, pad_length=1024):
    (X_train, y_train), _ = mnist.load_data()

    img_num = 10
    images = X_train[:img_num]
    labels = y_train[:img_num]

    images = [img.flatten() for img in images]
    images = [np.append(img, [1]) for img in images]

    if pad:
        images = [pad_to_length(img, pad_length) for img in images]

    for img in images:
        assert len(img) == 1024

    return list(np.array(images).flatten()), labels


def generate_coe():
    images, labels = get_mnist()
    coe_gen.generate_coe("vector.coe", images, labels, bytes_per_row=128)


if __name__ == "__main__":
    generate_coe()
