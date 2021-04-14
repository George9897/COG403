"""
This is the final Project - Automatic Number Plate Recognition
"""
from os import listdir
from matplotlib import pyplot as plt
import cv2
import numpy as np
from segmentation import character_segmentation
from localisation import localise_plate
from util import validate_char
from recognition import *
import sys
import os
show_grayscale = lambda x: plt.imshow(x, cmap="gray")
show_rgb = lambda x:plt.imshow(x)
show = plt.show


def get_images(directory_name="./images/", plot_flag=False):
    """
    Project main algorithm to invoke each process step by step on the images in the directory_name
    """
    # Load image directory
    image_list = sorted(filter(lambda x: x[0] != ".", listdir(directory_name)))[:]
    count = len(image_list)
    false_count = 0
    total_count, total_detected, total_recognized = 0, 0, 0

    # Read in validation file
    with open('validation.txt', 'r') as inf:
        validation_dict = eval(inf.read())

    # Iterate through directory
    for image_name in image_list[:]:
        image = os.path.join(directory_name, image_name)
        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        if plot_flag:
            plt.imshow(image)
            plt.show()
        # Plate localisation
        res = localise_plate(image, image_name, plot_flag)
        validate_set = validation_dict[image_name]
        if np.all(res == 0):
            false_count += 1
        # If a region was detected
        else:
            # Applied character segmentation
            total_count += len(validate_set)
            chars = character_segmentation(res, plot_flag)
            total_detected += validate_char(chars, validate_set, "seg")
            # Applied recognition on the characters obtained
            labels = predicate_char_cnn(chars, plot_flag=plot_flag)
            total_recognized += validate_char(labels, validate_set, "rec")

    # Calculate accuracy for each part
    print("The accuracy of localisation is among:{}".format((count-false_count) / count))
    msg = "The accuracy of character segmentation is {} out of {} (roughly {}% detection rate)"
    print(msg.format(total_detected, total_count, int(total_detected * 100 / total_count)))
    msg = "The number of character correctly recognize is {} out of {} (roughly {}% detection rate)"
    print(msg.format(total_recognized, total_detected, int(total_recognized * 100 / total_count)))


if __name__ == "__main__":
    args = sys.argv
    if len(args) >= 2:
        directory = args[1]
        plot = args[-1]
        if plot.upper() in ["TRUE", "FALSE"]:
            plot = True if plot.upper() == "TRUE" else False
            get_images(directory_name=os.path.join(".", directory), plot_flag=plot)
        else:
            get_images(directory_name=os.path.join(".", directory))
    else:
        get_images()
