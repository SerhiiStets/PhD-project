import os
import sys
import numpy as np
import cv2
import logging
import random

logger = logging.getLogger(__name__)

def merge_bounding_boxes(bounding_boxes, overlap_threshold=0.5):
    """
    A function to merge bounding boxes that overlap by a certain threshold.

    bounding_boxes: a list of tuples, where each tuple contains the x, y coordinates of the top-left corner of a bounding
                    box, as well as the width and height of the bounding box
    overlap_threshold: the threshold for overlapping bounding boxes to be merged

    returns: a list of merged bounding boxes
    """
    # create a list to store the merged bounding boxes
    merged_bounding_boxes = []

    # loop through the bounding boxes and merge overlapping ones
    for i in range(len(bounding_boxes)):
        # check if this bounding box has already been merged
        if bounding_boxes[i] is None:
            continue

        # initialize the merged bounding box parameters
        (x, y, w, h) = bounding_boxes[i]
        area = w * h

        for j in range(i + 1, len(bounding_boxes)):
            # check if this bounding box has already been merged
            if bounding_boxes[j] is None:
                continue

            # calculate the intersection over union (IoU) between the two bounding boxes
            (x2, y2, w2, h2) = bounding_boxes[j]
            area2 = w2 * h2

            dx = min(x + w, x2 + w2) - max(x, x2)
            dy = min(y + h, y2 + h2) - max(y, y2)

            if dx >= 0 and dy >= 0:
                intersection = dx * dy
                union = area + area2 - intersection
                iou = intersection / union

                # if the IoU is greater than the overlap threshold, merge the bounding boxes
                if iou >= overlap_threshold:
                    x = min(x, x2)
                    y = min(y, y2)
                    w = max(x + w, x2 + w2) - x
                    h = max(y + h, y2 + h2) - y
                    area = w * h

                    # mark the second bounding box as merged
                    bounding_boxes[j] = None

        # add the merged bounding box to the list of merged bounding boxes
        merged_bounding_boxes.append((x, y, w, h))

    return merged_bounding_boxes

def sliding_window(image, window_size, step_size, cnn):
    """
    A function to slide a window over an input image and detect cars in each window using a CNN.
    image: the input image to slide the window over
    window_size: a tuple containing the width and height of the window
    step_size: a tuple containing the horizontal and vertical step size of the window
    cnn: the trained CNN model for car detection
    """
    # initialize a list to store the bounding boxes
    bounding_boxes = []
    image_with_rectangles = image.copy()
    # slide the window over the image
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            # extract the current window from the image
            window = image[y : y + window_size[1], x : x + window_size[0]]
            window = cv2.resize(window, (100, 100))

            # draw a rectangle to show the current window
            cv2.rectangle(
                image_with_rectangles,
                (x, y),
                (x + window_size[0], y + window_size[1]),
                (0, 255, 0),
                2,
            )

            cv2.imshow("Current Window", window)
            cv2.imshow("Image with Rectangles", image_with_rectangles)
            # perform inference using the CNN on the current window
            prediction = cnn.predict(np.expand_dims(window, axis=0), verbose=0)
            # if an object is detected, draw a rectangle around the window
            if np.argmax(prediction) == 1:
                # draw a rectangle to show the current window
                cv2.rectangle(
                    image_with_rectangles,
                    (x, y),
                    (x + window_size[0], y + window_size[1]),
                    (255, 0, 0),
                    2,
                )
                bounding_boxes.append((x, y, window_size[0], window_size[1]))

            key = cv2.waitKey(200)

            if key == 27:  # if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                sys.exit(1)

            # show the image with the sliding window and a delay of 0.2 seconds

    # merge overlapping bounding boxes
    merged_boxes = merge_bounding_boxes(bounding_boxes)

    # draw the final bounding box around the car
    if len(merged_boxes) > 0:
        x, y, w, h = merged_boxes[0]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.destroyAllWindows()
    # display the final image with detected objects
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)

def sliding_window_search(images_dir: str, cnn):
    # максимальний розмів, зменишити в 2 рази, знову зменшити в 2 рази

    # load a random image from the 'cars' directory
    car_filenames = os.listdir(images_dir)
    random_car_filename = random.choice(car_filenames)
    logger.info(f"File: {random_car_filename}")
    car_path = os.path.join(images_dir, random_car_filename)
    car_image = cv2.imread(car_path)

    # TODO as variable m_w examples
    # TODO step to windows size (example,
    # if windows size is 400, step would be 1/4 of windows size)

    # set the window size and step size
    window_size = (400, 400)
    step_size = (100, 100)

    # call the sliding_window function
    sliding_window(car_image, window_size, step_size, cnn)
