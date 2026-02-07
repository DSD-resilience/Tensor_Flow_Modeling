# train.py

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

# ---- Configuration ----
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'models/model.h5'


# ---- Load and Prepare Data ----
def load_data():
    """
    Replace with custom logic if loading from CSV.
    This example uses MNIST from TensorFlow datasets.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Add a channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return x_train, x_test, y_train, y_test


# ---- Build Model ----
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ---- Training ----
def train():
    print("Loading data...")
    x_train, x_test, y_train, y_test = load_data()
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

    print("Building model...")
    model = build_model(input_shape, num_classes)

    print("Starting training...")
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

    print("Evaluating model...")
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")


# ---- Entry Point ----
if __name__ == "__main__":
    train()
