import tensorflow as tf

import utilities
import data

MODEL = "./model.ckpt"


def save_model(model, filename=MODEL):
    """ Saves the model. """
    model.save(filename)


def load_model(filename=MODEL):
    """ Loads the model file. """
    return tf.keras.models.load_model(filename)


def train():
    """ Trains the model. """
    x_train, y_train = data.load_training_data()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=utilities.IMG_SIZE))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(6, activation=tf.nn.softmax))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=["accuracy"])

    print(model.summary())

    model.fit(x_train, y_train, epochs=30)

    return model


def main():
    """ Train the model based on the image data from the data file. """
    model = train()
    save_model(model)
    print("Model saved")


if __name__ == "__main__":
    main()
