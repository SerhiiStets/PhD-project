import os

import cv2
import numpy as np
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D, concatenate)
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from skimage import transform
from sklearn.metrics import accuracy_score


class DataLoader:
    def __init__(
        self,
        car_images_dir: str,
        not_car_images_dir: str,
    ) -> None:
        self.car_images_dir = car_images_dir
        self.not_car_images_dir = not_car_images_dir

    def load_images_from_dir(self, args):
        images_dir, label = args
        images = os.listdir(images_dir)
        train_images = []
        train_labels = []
        for image in images:
            if image.endswith("jpeg") or image.endswith("jpg"):
                path = os.path.join(images_dir, image)
                img = cv2.imread(path)
                new_img = self.reshaped_image(img)
                train_images.append(new_img)
                l = [0, 0]
                l[label] = 1  # 1=car and 0=not car
                train_labels.append(l)
        return train_images, train_labels

    def load_images(self):
        train_images = []
        train_labels = []

        images, labels = self.load_images_from_dir((self.car_images_dir, 1))
        train_images.extend(images)
        train_labels.extend(labels)
        images, labels = self.load_images_from_dir((self.not_car_images_dir, 0))
        train_images.extend(images)
        train_labels.extend(labels)

        return np.array(train_images), np.array(train_labels)

    def reshaped_image(self, image: np.array) -> np.array:
        h, w, c = image.shape
        if h == w:
            # Image is already square, resize and return
            return transform.resize(image, (100, 100, c))
        elif h > w:
            # Image is vertically oriented, pad left and right
            pad_size = (h - w) // 2
            left_pad = np.mean(image[:, :pad_size, :], axis=(0, 1), keepdims=True)
            right_pad = np.mean(image[:, -pad_size:, :], axis=(0, 1), keepdims=True)
            padded_image = np.pad(
                image,
                ((0, 0), (pad_size, pad_size), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            padded_image[:, :pad_size, :] = left_pad
            padded_image[:, -pad_size:, :] = right_pad
        else:
            # Image is horizontally oriented, pad top and bottom
            pad_size = (w - h) // 2
            top_pad = np.mean(image[:pad_size, :, :], axis=(0, 1), keepdims=True)
            bottom_pad = np.mean(image[-pad_size:, :, :], axis=(0, 1), keepdims=True)
            padded_image = np.pad(
                image,
                ((pad_size, pad_size), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            padded_image[:pad_size, :, :] = top_pad
            padded_image[-pad_size:, :, :] = bottom_pad

        return transform.resize(padded_image, (100, 100, c))

    def normalize_images_for_cnn(self, images):
        # Convert to float32 and normalize to 0-1
        images = images.astype("float32") / 255.0

        # Subtract the mean pixel value across all images
        mean = np.mean(images, axis=0)
        images -= mean

        # Divide by the standard deviation across all images
        std = np.std(images, axis=0)
        images /= std
        return images


car_images_dir: str = "./data/cars/"
not_car_images_dir: str = "./data/not_cars/"
object_detection_dir = "./data/object_detection/"
data_loader = DataLoader(car_images_dir, not_car_images_dir)

train_data, train_labels = data_loader.load_images()


# Define input shape
input_shape = (100, 100, 3)

# # Create a sequential model
# model = Sequential()
#
# # First Convolutional Layer
# model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# model.add(Dropout(0.25))
#
# # Second Convolutional Layer
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# model.add(Dropout(0.25))
#
# # Third Convolutional Layer
# model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# model.add(Dropout(0.25))
#
# # Flatten layer
# model.add(Flatten())
#
# # Fully connected layers
# model.add(Dense(128, activation='relu'))
# model.add(Dense(2, activation='softmax'))  # 2 outputs for binary classification
#
# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Print the model summary
# model.summary()
#
#
# # Define input layer with smaller input shape
# input_layer = Input(shape=(100, 100, 3))
#
# # First Inception module with reduced filter sizes and dimensions
# conv1x1_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)
#
# conv3x3_reduce_1 = Conv2D(48, (1, 1), padding='same', activation='relu')(input_layer)
# conv3x3_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv3x3_reduce_1)
#
# conv5x5_reduce_1 = Conv2D(8, (1, 1), padding='same', activation='relu')(input_layer)
# conv5x5_1 = Conv2D(16, (5, 5), padding='same', activation='relu')(conv5x5_reduce_1)
#
# maxpool_1 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(input_layer)
# maxpool_proj_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(maxpool_1)
#
# # Concatenate the outputs of the filters
# inception_1 = concatenate([conv1x1_1, conv3x3_1, conv5x5_1, maxpool_proj_1], axis=-1)
#
# # Flatten layer
# flatten = Flatten()(inception_1)
#
# # Fully connected layers
# dense1 = Dense(128, activation='relu')(flatten)
# output_layer = Dense(2, activation='softmax')(dense1)  # 2 outputs for binary classification
#
# model = Model(inputs=input_layer, outputs=output_layer)
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# model.summary()
#

# function for creating a vgg block
def vgg_block(layer_in, n_filters, n_conv):
 # add convolutional layers
 for _ in range(n_conv):
     layer_in = Conv2D(n_filters, (3,3), padding='same', activation='relu')(layer_in)
     # add max pooling layer
 layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
 return layer_in
#  
# # define model input
# visible = Input(shape=input_shape)
# # add vgg module
# layer = vgg_block(visible, 32, 1)
# # add vgg module
# layer = vgg_block(layer, 64, 1)
# # add vgg module
# layer = vgg_block(layer, 16, 1)
#
# flatten = Flatten()(layer)
# dense1 = Dense(128, activation='relu')(flatten)
# output_layer = Dense(2, activation='softmax')(dense1)  # 2 outputs for binary classification
# # create model
# model = Model(inputs=visible, outputs=output_layer)
#

# function for creating a naive inception block
def naive_inception_module(layer_in, f1, f2, f3):
    # 1x1 conv
    conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
    # 3x3 conv
    conv3 = Conv2D(f2, (3,3), padding='same', activation='relu')(layer_in)
    # 5x5 conv
    conv5 = Conv2D(f3, (5,5), padding='same', activation='relu')(layer_in)
    # 3x3 max pooling
    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out
 
# define model input
visible = Input(shape=(100, 100, 3))
# # add inception module
layer = naive_inception_module(visible, 32, 64, 16)
layer = naive_inception_module(layer, 32, 64, 16)
flatten = Flatten()(layer)
dense1 = Dense(128, activation='relu')(flatten)
output_layer = Dense(2, activation='softmax')(dense1)  # 2 outputs for binary classification
# create model
model = Model(inputs=visible, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def train_test_split(
    train_data: np.ndarray, train_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The function shuffles the input data and labels using the same random permutation, and then splits the data
    and labels into train and test sets according to the specified fraction. The split is performed along the first
    dimension of the arrays. The function returns the shuffled train and test data and labels as numpy arrays.
    """
    # Shuffle the indices
    indices = list(range(train_data.shape[0]))
    import random

    random.shuffle(indices)

    # Compute the split index
    split_index = int(train_data.shape[0] * 0.7)
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


(
    train_data,
    train_labels,
    test_data,
    test_labels,
) = train_test_split(train_data, train_labels)


early_stopping = EarlyStopping(monitor='val_loss', patience=3)


history = model.fit(
    train_data,
    train_labels,
    validation_data=(test_data, test_labels),
    epochs=10,
    callbacks=[early_stopping]
)

predicted_test_labels = np.argmax(model.predict(test_data), axis=1)
test_labels = np.argmax(test_labels, axis=1)
print(f"Predicted test labels: {predicted_test_labels[0:10]}")
print(f"Actual test labels: {test_labels[0:10]}")
print(f"Accuracy score: {accuracy_score(test_labels, predicted_test_labels)}")
