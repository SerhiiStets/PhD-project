import logging
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import models
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

class CNNModel:
    def __init__(
        self,
        car_images_dir: str = "./cars/",
        not_car_images: str = "./not_cars/",
        split_fraction: float = 0.8,
    ) -> None:
        self.cnn = None
        self.history = None
        self.split_fraction = split_fraction
        self.car_images_dir = car_images_dir
        self.not_car_images_dir = not_car_images

        # Check that the fraction is valid
        if self.split_fraction < 0 or self.split_fraction > 1:
            raise ValueError(
                "Invalid fraction: {}. Fraction must be between 0 and 1.".format(
                    self.split_fraction
                )
            )

    def _train_test_split(
        self, train_data: np.ndarray, train_labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        The function shuffles the input data and labels using the same random permutation, and then splits the data
        and labels into train and test sets according to the specified fraction. The split is performed along the first
        dimension of the arrays. The function returns the shuffled train and test data and labels as numpy arrays.
        """
        # Shuffle the indices
        indices = list(range(train_data.shape[0]))

        random.shuffle(indices)

        # Compute the split index
        split_index = int(train_data.shape[0] * self.split_fraction)
        # Split the data and labels using the shuffled indices
        train_data_shuffled = train_data[indices[:split_index]]
        train_labels_shuffled = train_labels[indices[:split_index]]
        test_data_shuffled = train_data[indices[split_index:]]
        test_labels_shuffled = train_labels[indices[split_index:]]

        return (
            train_data_shuffled,
            train_labels_shuffled,
            test_data_shuffled,
            test_labels_shuffled,
        )

    def build_cnn(self) -> Sequential:
        cnn = Sequential()
        cnn.add(
            Conv2D(
                8, (3, 3), input_shape=(100, 100, 3), padding="same", activation="relu"
            )
        )
        cnn.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        cnn.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
        cnn.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        cnn.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
        cnn.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        cnn.add(Flatten())
        cnn.add(Dense(128, activation="relu"))
        cnn.add(Dense(2, activation="softmax"))  # 2 outputs
        # TODO: batch nomalization
        # TODO: Global Max Pooling layer
        # TODO: sparce categorical cross entropy
        cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        logger.info(cnn.summary())
        return cnn

    def run_model(
        self, train_data, train_labels
    ):
        self.train_data, self.train_labels = train_data, train_labels
        (
            self.train_data,
            self.train_labels,
            self.test_data,
            self.test_labels,
        ) = self._train_test_split(self.train_data, self.train_labels)
        logger.info(f"Train data size: {len(self.train_data)}")
        logger.info(f"Test data size: {len(self.test_data)}")

        self.cnn = self.build_cnn()
        self.history = self.cnn.fit(
            self.train_data,
            self.train_labels,
            validation_data=(self.test_data, self.test_labels),
            epochs=10,
        )

        logger.info(f"Train data shape: {self.train_data.shape}")
        logger.info(f"Test data shape: {self.test_data.shape}")

    def prediction(self, show_incorrect_indices: bool = False):
        self.predicted_test_labels = np.argmax(self.cnn.predict(self.test_data), axis=1)
        self.test_labels = np.argmax(self.test_labels, axis=1)
        logger.info(f"Predicted test labels: {self.predicted_test_labels[0:10]}")
        logger.info(f"Actual test labels: {self.test_labels[0:10]}")
        logger.info(
            f"Accuracy score: {accuracy_score(self.test_labels, self.predicted_test_labels)}"
        )
        self.incorrect_indices = np.nonzero(
            self.predicted_test_labels != self.test_labels
        )[0]
        logger.info(f"Number of incorrect predictions: {len(self.incorrect_indices)}")
        logger.info(
            f"Incorrectly predicted test labels: {self.predicted_test_labels[self.incorrect_indices]}"
        )
        logger.info(f"Actual test labels: {self.test_labels[self.incorrect_indices]}")


    def load_model(self, path: str):
        self.cnn = models.load_model(path)
        

    def save_model(self, path: str):
        if self.cnn:
            self.cnn.save(path)
