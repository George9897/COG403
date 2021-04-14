"""
This file contains all the helper functions utilized
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def morphology_kernel(height, weight):
    """
    Helper function to return morphology kernel
    """
    return np.ones((height, weight))


def dilation(image, kernel):
    """
    Helper function to applied dilation to the image using kernel
    """
    res = image.copy()
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            height_in_range = (h - (kernel_height // 2) >= 0) and (h + (kernel_height // 2) < image_height)
            width_in_range = (w - (kernel_width // 2) >= 0) and (w + (kernel_width // 2) < image_width)
            if height_in_range and width_in_range:
                neighbours = image[h-(kernel_height // 2):h + (kernel_height // 2) + 1, w - (kernel_width // 2): w + (kernel_width // 2) + 1]
                res[h, w] = neighbours.max()
    return res


def erosion(image, kernel):
    """
    Helper function for applied erosion to the image using kernel
    """
    res = image.copy()
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    for h in range(image.shape[0]):
        for w in range(image.shape[1]):
            height_in_range = (h - (kernel_height // 2) >= 0) and (h + (kernel_height // 2) < image_height)
            width_in_range = (w - (kernel_width // 2) >= 0) and (w + (kernel_width // 2) < image_width)
            if height_in_range and width_in_range:
                neighbours = image[h - (kernel_height // 2):h + (kernel_height // 2) + 1, w - (kernel_width // 2): w + (kernel_width // 2) + 1]
                res[h, w] = neighbours.min()
    return res


def sort_contours(contours, reverse=False):
    """
    Create sort_contours() function to grab the contour of each digit from left to right
    """
    i = 0
    # Unpack detected contours to a list
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    # Sort based on the left index location of the boxes detected
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return contours


def add_padding(letter, pad_w):
    """
    Add padding to the detected binary letter image for better readability
    """
    h, w = letter.shape
    # Initiate resulting array with according size
    result = np.zeros((h + 2 * pad_w, w + 2 * pad_w))
    # Iterating through original image
    for i in range(h):
        for j in range(w):
            # Add corresponding pixel value to result image starting at the index of the padding width and height
            result[i + pad_w, j + pad_w] = letter[i, j]
    return result


def compare_contours(existing, box):
    """
    Compare the current box region with all previous qualified regions to eliminate duplicated inner contours
    """
    # For first letter in each plate
    if len(existing) == 0:
        return True
    # Iterate through all existing regions
    for box_item in existing:
        (x1, y1, w1, h1) = box_item
        (x2, y2, w2, h2) = box
        # In the condition of starting index located in the range of any previous regions
        if (x1 + w1) > x2 and (y1 + h1) > y2:
            # In the condition of overlapping length is over 10%
            if (x1 + w1 - x2) / w2 > 0.1:
                return False
    # If nothing overlapping found, return True
    return True


def save_fig(img, count):
    """
    Helper function for generating additional training dataset
    """
    # Configure saved figure size, output to dimension 30*60
    plt.figure(figsize=(45, 78), dpi=1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    # Save to directory
    plt.savefig('./letters/'+str(count)+".jpg", bbox_inches='tight', pad_inches=0)


def in_range(lower_bound, val, upper_bound, exclusive=True):
    """
    Helper function to check if val is in between lower and upper bound whether exclusive or not
    """
    if not exclusive:
        return lower_bound <= val <= upper_bound
    else:
        return lower_bound < val < upper_bound


def validate_char(results, compare, mode):
    """
    Helper function used for validating segmentation or recognition results
    """
    count = 0
    lst = compare.copy()
    if len(results) == 0 or len(compare) == 0:
        return count
    # Validation for character segmentation mode
    if mode == "seg":
        count = len(results) if len(results) <= len(compare) else len(compare)
    # Validation for character recognition mode
    elif mode == "rec":
        checked = set()
        for letter in results:
            if letter in compare:
                ind = compare.index(letter)
                compare[ind] = -1
                if ind not in checked:
                    checked.add(ind)
                    count += 1
    return count


def plot_images(images, titles, axis_off=True):
    def imshow(image):
        if len(image.shape) == 3:
            plt.imshow(image)
        else:
            plt.imshow(image, cmap="gray")
    number_of_images = len(images)
    for i in range(len(images)):
        plt.subplot(1, number_of_images, (i + 1))
        plt.title(titles[i])
        if axis_off:
            plt.axis("off")
        imshow(images[i])
    plt.show()
