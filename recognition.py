"""
this file is used to recognize the character separated from the number plate, each function takes the segmented characters
as its argument and return the predicted characters by the model.
"""
import pickle
import cv2
from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu


def predicate_char_cnn(nums, img_rows=30, img_cols=60, plot_flag=False):
    """
    predict the character which is separated from the number plate using convolutional neural network
    """
    numchar = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
               'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    model = load_model('data_char.keras')
    char_img = []
    label = []
    # nums are list of character segmented
    for ch in nums:
        ch = ch.reshape(ch.shape).astype('float32')
        each_character = cv2.resize(ch, (img_rows, img_cols))
        image = each_character.reshape((1, img_rows, img_cols, 1))
        new_predictions = model.predict(image)
        char_index = int(np.argmax(new_predictions))
        char = numchar[char_index]
        char_img.append(ch)
        label.append(char)
    if plot_flag:
        fig = plt.figure()
        for i in range(len(char_img)):
            a = fig.add_subplot(1, len(char_img), i + 1)
            plt.imshow(char_img[i], cmap="gray")
            a.title.set_text(str(label[i]))
            a.axis("off")
        plt.show()
    return label


def predicate_char_dt(crop_character, img_rows=784, img_cols=1):
    """
    predict the character which is separated from the number plate using decision tree classifier
    """
    model_file = 'decision_tree.sav'
    model = pickle.load(open(model_file, 'rb'))
    char_img = []
    label = []
    for ch in crop_character:
        each_character = cv2.resize(ch, (img_rows, img_cols))
        new_predictions = model.predict(each_character)
        char_img.append(ch)
        label.append(new_predictions)
    fig = plt.figure()
    for i in range(len(char_img)):
        a = fig.add_subplot(1, len(char_img), i + 1)
        plt.imshow(char_img[i], cmap="gray")
        a.title.set_text(str(label[i]))
        a.axis("off")
    # plt.show()
    return label


def predicate_char_svc(crop_character, img_rows=1800, img_cols=1):
    """
    predict the character which is separated from the number plate using decision tree classifier
    """
    model_file = 'svc.sav'
    model = pickle.load(open(model_file, 'rb'))
    char_img = []
    label = []
    for ch in crop_character:
        each_character = cv2.resize(ch, (img_rows, img_cols))
        binary_image = each_character < threshold_otsu(each_character)
        flattened_image = binary_image.reshape(-1)
        flattened_image = [flattened_image]
        new_predictions = model.predict(flattened_image)
        char_img.append(ch)
        label.append(new_predictions)
    fig = plt.figure()
    for i in range(len(char_img)):
        a = fig.add_subplot(1, len(char_img), i + 1)
        plt.imshow(char_img[i], cmap="gray")
        a.title.set_text(str(label[i]))
        a.axis("off")
    # plt.show()
    return label


def predicate_char_knn(crop_character, img_rows=784, img_cols=1):
    """
    predict the character which is separated from the number plate using K-nearest neighbor classifier
    """
    model_file = 'knn.sav'
    model = pickle.load(open(model_file, 'rb'))
    char_img = []
    label = []
    for ch in crop_character:
        each_character = cv2.resize(ch, (img_rows, img_cols))
        new_predictions = model.predict(each_character)
        char_img.append(ch)
        label.append(new_predictions)
    fig = plt.figure()
    for i in range(len(char_img)):
        a = fig.add_subplot(1, len(char_img), i + 1)
        plt.imshow(char_img[i], cmap="gray")
        a.title.set_text(str(label[i]))
        a.axis("off")
    # plt.show()
    return label
