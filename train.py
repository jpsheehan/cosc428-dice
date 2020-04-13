import os
import tensorflow as tf
import numpy as np
import cv2

import utilities

DIR_NAME = "./die_images/"
MODEL = "./model.ckpt"


def get_filename_value_pairs():
    data = []
    for filename in os.listdir(DIR_NAME):
        parts = filename.split("_")
        if len(parts) == 3:
            full_filename = os.path.join(DIR_NAME, filename)
            value = int(parts[0])

            if value > 0 and value <= 6:
                data.append((full_filename, value))
    return data


def load_classified_data():
    pairs = get_filename_value_pairs()
    # read the images in, converting them to grayscale
    x = np.array([cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                  for filename, _value in pairs])
    # get the actual values, they must be zero based, so 0 really means 1, etc
    y = np.array([value - 1 for _filename, value in pairs])
    return x, y


def save_model(model, filename=MODEL):
    model.save(filename)


def load_model(filename=MODEL):
    return tf.keras.models.load_model(filename)


def train():
    x_train, y_train = load_classified_data()

    x_train = x_train / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=utilities.IMG_SIZE),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6)
    ])

    # predictions = model(x_train[:1]).numpy()
    # print("Predictions:", predictions)

    # probs = tf.nn.softmax(predictions).numpy()
    # print("Probabilities:", probs)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # loss_fn(y_train[:1], predictions).numpy()

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20)

    # model = tf.keras.Sequential([
    #     model,
    #     tf.keras.layers.Softmax()
    # ])

    return model


def main():
    model = train()
    save_model(model)
    print("Model saved")


if __name__ == "__main__":
    main()
