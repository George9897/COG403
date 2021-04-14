"""
COG403 - Plate Localization Algorithm Code
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from util import morphology_kernel, dilation, erosion, plot_images


def localise_plate(image, image_name, plot=False):
    """
    Plate localisation algorithm to detect plate region from the image
    """
    print("Image:{}".format(image_name))
    # Resize the image to 640 & 480
    image_copy = image.copy()
    # Convert rgb array to gray scale
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Blur the image
    image = cv2.GaussianBlur(image_gray, (9, 9), 3)
    if plot:
        plot_images([image_copy, image_gray, image], ["Original Image", "Grayscale Image", "Image Blurring"])

    # Compute Directional Gradient and Gradient Magnitude
    Gx, Gy = cv2.Sobel(image, cv2.CV_8U, 1, 0), cv2.Sobel(image, cv2.CV_8U, 0, 1)
    M = np.abs(Gx) + np.abs(Gy)
    if plot:
        plot_images([Gx, Gy, M], ["X gradient", "Y Gradient", "Image Gradient"])

    # Threshold the image
    _, image = cv2.threshold(M, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = morphology_kernel(5, 23)
    if plot:
        dilate = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)
        close = cv2.morphologyEx(dilate, cv2.MORPH_ERODE, kernel)
        plot_images([image, dilate, close], ["Threshold Image", "Dilation", "Erosion"])
    # Apply morphological closing to the image
    image_closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # Get contours in the image
    _, contours, hierarchy = cv2.findContours(image_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    candidates = []
    # Finding plate candidates
    for hull in contours[:4]:
        (x, y, w, h) = cv2.boundingRect(hull)
        ratio = h/w
        candidate = image_copy[y:y + h, x:x + w]
        if h in range(35, 110) and 0.1 <= ratio < 0.31:
            candidates.append((candidate, ratio))

    # Filter Candidates
    if not candidates:
        return np.zeros((300, 100))
    else:
        # Iterate all contours in the candidate
        # Find out the one with the largest ratio
        best_ratio = -np.inf
        res = None
        for i in range(len(candidates)):
            curr, ratio = candidates[i]
            if res is None or ratio > best_ratio:
                res = curr
                best_ratio = ratio
        if plot:
            plt.title("Plate")
            plt.imshow(res)
            plt.show()
        return res
