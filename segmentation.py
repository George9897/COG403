"""
Character Segmentation Algorithm Code
"""
import cv2
import matplotlib.pyplot as plt
from util import sort_contours, add_padding, compare_contours, dilation, erosion, in_range, plot_images


def character_segmentation(img, plot_flag=False):
    """
    Main function for character segmentation, helper function sort_contours; compare_contours will be called
    """
    # Ensure image passed in is not empty
    if img is not None:
        # convert to grayscale and blur the image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Applied threshold to convert to binary image
        binary = cv2.threshold(gray, 10, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Applied dilation and erode process to binary
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # dilate_image = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)
        # closed = cv2.morphologyEx(dilate_image, cv2.MORPH_ERODE, kernel)
        dilate_image = dilation(binary, kernel)
        closed = erosion(dilate_image, kernel)

        # Find contours using built in function
        _, cont, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Use a copy version of plat_image to draw bounding box (all boxes and filtered)
        test_roi = img.copy()
        passed_in = img.copy()

        # Initialize empty list which for appending detected character image
        crop_characters = []

        # define tempting return standard width and height of character (Should be tried for different size)
        digit_w, digit_h = 30, 60

        # Find area of interest by using proportion of input plate width and height
        min_height, max_height, min_width, max_width = (0.26 * img.shape[0], 0.78 * img.shape[0], 0.014 * img.shape[1], 0.12 * img.shape[1])

        # Loop over sorted contours
        contour_num, existing = 0, []
        for c in sort_contours(cont):
            msg = ""
            contour_num += 1
            (x, y, w, h) = cv2.boundingRect(c)
            box = tuple((x, y, w, h))
            ratio = h / w
            # print("Contour height {} ({}), width {} ({})".format(h, h/img.shape[0], w, w/img.shape[1]))
            cv2.rectangle(passed_in, (x, y), (x + w, y + h), (0, 255, 0), 1)
            if in_range(min_height, h, max_height) and in_range(min_width, w, max_width):  # Only select contour with qualified height width
                if in_range(1, ratio, 9, exclusive=False):  # Only select contour with defined ratio
                    if abs(h - w) >= 6:  # Select contour which has the height width difference >= 6
                        if compare_contours(existing, box):  # No previous overlapping or duplicate contours found
                            msg += "No previous contours detected! Want this box!"
                            # Draw rectangles around detected region
                            cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 0, 255), 2)

                            # Extract character region; then resize to standard size and convert to binary image for the
                            # result; Update qualifying contours for future checking
                            _, curr_num = cv2.threshold(closed[y:y + h, x:x + w], 230, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            existing.append((x, y, w, h))

                            # Add padding to the detected region for better readability for machine
                            added = cv2.resize(add_padding(curr_num, 5), dsize=(digit_w, digit_h))

                            # Add to result list
                            crop_characters.append(added)

        print("Detect {} letters out of {} contours...".format(len(crop_characters), contour_num))
        # Plot is enabled when wanted
        if plot_flag is True:
            plot_images([passed_in, test_roi], ["All contours", "Detected Letters"])
        return crop_characters
    else:
        print("Image passed in is empty; should not reach here!!!")
