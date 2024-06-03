# for data load
import os

# for reading and processing images
import imageio
from PIL import Image

# for visualizations
import matplotlib.pyplot as plt

import numpy as np # for using np arrays

# For splitting dataset to train, test.
from sklearn.model_selection import train_test_split

import tensorflow as tf

from U_net_model import UNetCompiled
from Helper_Functions import load_and_process


def Train(display, n):
    """ Load Train Set and view some examples """
    img, mask = load_and_process(display,n)

    X_train, X_valid, y_train, y_valid = train_test_split(np.array(img), np.array(mask), test_size=0.2, random_state=123)

    print(f"Train X size : {len(X_train)} , Test X size : {len(X_valid)}, Train Y size : {len(y_train)}, Test Y size : {len(y_valid)}")
    

    # Initializing the Unet model
    model = UNetCompiled(input_size=(512,512,1), n_filters=32, n_classes=2)

    print("Model Summary : ")
    print(model.summary())
    

    # There are multiple optimizers, loss functions and metrics that can be used to compile multi-class segmentation models
    # Ideally, try different options to get the best accuracy
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    # Run the model in a mini-batch fashion and compute the progress for each epoch
    results = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_valid, y_valid))

    model.evaluate(X_valid, y_valid)

    output_path = "./Weights/"
    model.save_weights(output_path)

if __name__ == "__main__":
    Train(False, 5)