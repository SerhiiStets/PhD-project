import logging
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np
from skimage import transform

logger = logging.getLogger(__name__)


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
                # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.imread(path)
                new_img = self.reshaped_image(img)
                # if "1516" in image:
                # cv2.imshow("img1", img)
                # cv2.imshow("image", new_img)
                # cv2.waitKey(0)
                train_images.append(new_img)
                l = [0, 0]
                l[label] = 1  # 1=car and 0=not car
                train_labels.append(l)
        return train_images, train_labels

    def load_images(self):
        train_images = []
        train_labels = []

        logger.info("Start loading the images")
        start_time = time.time()

        with Pool() as p:
            results = p.map(
                self.load_images_from_dir,
                [(self.car_images_dir, 1), (self.not_car_images_dir, 0)],
            )

        for images, labels in results:
            train_images.extend(images)
            train_labels.extend(labels)

        end_time = time.time() - start_time
        logger.info(f"End loading the images. Time: {end_time}")
        return np.array(train_images), np.array(train_labels)

    def reshaped_image(self, image: np.array) -> np.array:
        """
        The reshaped_image function takes an image as input and returns a reshaped version of the image.
        The reshaped image is either a 100x100 square version of the original image, or a padded version
        of the original image where the padding is added to the shorter side of the image. The padding is
        added in a way to maintain the aspect ratio of the original image, and the padded area is filled
        with the average color of the top or left side (for vertical or horizontal orientation, respectively)
        or the bottom or right side (for vertical or horizontal orientation, respectively).
        """
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
        """
        Normalize the input images for CNN training by subtracting the mean
        pixel value and dividing by the standard deviation.
        The _normalize_images_for_cnn function takes a list or an array of
        images as input and returns a normalized version of the images that
        can be used as input for a convolutional neural network (CNN).
        """
        logger.info("Start normalizing the images")
        # Convert to float32 and normalize to 0-1
        images = images.astype("float32") / 255.0

        # Subtract the mean pixel value across all images
        mean = np.mean(images, axis=0)
        images -= mean

        # Divide by the standard deviation across all images
        std = np.std(images, axis=0)
        images /= std
        logger.info("End normalizing the images")
        return images
