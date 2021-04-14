"""
COG403 Model Training Algorithm
the CNN model is refer to https://missinglink.ai/guides/convolutional-neural-networks/python-convolutional-neural-network-creating-cnn-keras-tensorflow-plain-python/
"""

import os
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import layers
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import load_model
from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split
import glob
import pickle
import numpy as np
import cv2
from keras.models import Sequential


def load_data():
    """
    load all the images for training by the KNN and the decision tree classifier
    """
    img_rows, img_cols = 784, 1
    np.random.seed(30)
    # get path to all the images
    dataset_paths = glob.glob("dataset_characters/**/*.jpg")
    # list of images
    x = []
    # list of corresponding labels
    y = []
    for image_path in dataset_paths[::]:
        label = image_path.split(os.path.sep)[-2]
        img_num = image_path.split(os.path.sep)[-1]
        image_path = os.path.join('./dataset_characters', str(label), str(img_num))
        img1 = cv2.imread(image_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = cv2.resize(img1, (img_rows, img_cols))
        x.append(img1[0])
        y.append(label)
    x = np.asarray(x)
    y = np.asarray(y)
    # split two arrays into 80% training and 20% test subsets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # split two arrays into half validation and half test subsets
    x_validation, x_testex, y_validation, y_testex = train_test_split(x_test, y_test, train_size=0.5, test_size=0.5)
    return x_train, y_train, x_validation, x_testex, y_validation, y_testex


def select_tree_model(news_train, indicator_train, news_validation, indicator_validation,
                      news_testex, indicator_testex):
    """
    return the decision tree classifier with highest accuarcy by tuning the hyperparameters
    """
    result = ['', 'entropy', 60, 0]
    # tune hyperparameters criteria and depth, choose the highest accuracy ones to fit the decision tree model
    for criteria in ["gini", "entropy"]:
        for depth in range(1, 30):
            count = 0
            dt = DecisionTreeClassifier(criterion=criteria, max_depth=depth)
            dt.fit(news_train, indicator_train)
            prediciton = dt.predict(news_validation)
            # compute the validation accuracy
            for i in range(len(prediciton)):
                if str(prediciton[i]) == str(indicator_validation[i]):
                    count += 1
            accuarcy = count / len(prediciton)
            if accuarcy > result[2]:
                result = [dt, criteria, depth, accuarcy]
    # model with highest validation accuracy
    highest = DecisionTreeClassifier(criterion=result[1], max_depth=result[2])
    count_test = 0
    highest.fit(news_train, indicator_train)
    prediciton2 = highest.predict(news_testex)
    # compute the test accuracy
    for i in range(len(prediciton2)):
        if str(prediciton2[i]) == str(indicator_testex[i]):
            count_test += 1
    accuarcy_test = count_test / len(prediciton2)
    result[-1] = accuarcy_test
    return result


def construct_dt():
    """
    construct the decision tree model
    """
    x_train, y_train, x_validation, x_testex, y_validation, y_testex = load_data()
    # best decision tree model
    best_model = select_tree_model(x_train, y_train,
                                   x_validation, y_validation, x_testex, y_testex)
    _, criterion, max_depth, accuracy = best_model
    dtc = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    # fit and save the decision tree model
    dtc.fit(x_train, y_train)
    file_dt = "decision_tree.sav"
    pickle.dump(dtc, open(file_dt, 'wb'))
    print("decision tree model saved !")


def select_knn_model(news_train, indicator_train, news_validation,
                     indicator_validation, news_testex, indicator_testex):
    """
    return the KNeighborsClassifier with highest accuarcy by tuning the hyperparameters
    """
    result = ('', 0, 0)
    # tune hyperparameter k
    for i in range(1, 10):
        count_val = 0
        knn_model = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
        knn_model.fit(news_train, indicator_train)
        prediciton_val = knn_model.predict(news_validation)
        for k in range(len(prediciton_val)):
            if str(prediciton_val[k]) == str(indicator_validation[k]):
                count_val += 1
        accuarcy_val = count_val / len(prediciton_val)
        if accuarcy_val > result[1]:
            result = (knn_model, i, accuarcy_val)
    # choose model with highest validation accuracy
    count_test = 0
    knn_model = KNeighborsClassifier(n_neighbors=result[1])
    knn_model.fit(news_train, indicator_train)
    # compute the test accuarcy
    prediciton_test = knn_model.predict(news_testex)
    for m in range(len(prediciton_test)):
        if str(prediciton_test[m]) == str(indicator_testex[m]):
            count_test += 1
    accuarcy_test = count_test / len(prediciton_test)
    print("best model with k = " + str(result[1]) + " : accuracy on the test data is " + str(accuarcy_test))
    return knn_model


def construct_knn():
    """
    construct the knn model and save it to the file knn.sav
    """
    x_train, y_train, x_validation, x_testex, y_validation, y_testex = load_data()
    # best knn model obtained
    k_model = select_knn_model(x_train, y_train, x_validation, y_validation, x_testex, y_testex)
    # save the knn model
    file_knn = 'knn.sav'
    pickle.dump(k_model, open(file_knn, 'wb'))
    print("KNN model saved !")


def one_hot_encoding(all_char, y_train, y_test):
    """
    one hot encoding the label array
    """
    y_train_result = []
    y_test_result = []
    # convert label into a list of 0s and 1
    for i in range(len(y_train)):
        result = [0] * len(all_char)
        char = y_train[i]
        ind = all_char.index(char)
        result[ind] = 1
        result = np.asarray(result)
        y_train_result.append(result)
    # repeat the process on the test label array
    for i in range(len(y_test)):
        result = [0] * len(all_char)
        char = y_test[i]
        ind = all_char.index(char)
        result[ind] = 1
        result = np.asarray(result)
        y_test_result.append(result)
    y_train_result = np.asarray(y_train_result)
    y_test_result = np.asarray(y_test_result)
    return y_train_result, y_test_result


def construct_cnn(train=False):
    """
    construct the convolutional neural network(CNN) model
    """
    all_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    img_rows, img_cols = 30, 60
    np.random.seed(30)
    # path to all the images
    dataset_paths = glob.glob("dataset_characters/**/*.jpg")
    x = []
    y = []
    # load images
    for image_path in dataset_paths[::]:
        label = image_path.split(os.path.sep)[-2]
        img_num = image_path.split(os.path.sep)[-1]
        image_path = os.path.join('./dataset_characters', str(label), str(img_num))
        img1 = cv2.imread(image_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = cv2.resize(img1, (img_rows, img_cols))
        x.append(img1)
        y.append(label)
    x = np.asarray(x)
    y = np.asarray(y)
    # split into 75% training and 25% test subsets
    x_train, x_test, y_train1, y_test1 = train_test_split(x, y, test_size=0.25)
    # reshape the image arrays in order to be an input of CNN model, 1 represents grayscale
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    # one hot encoding the label arrays
    y_train, y_test = one_hot_encoding(all_char, y_train1, y_test1)
    # create a Sequential model, refer to https://missinglink.ai/guides/convolutional-neural-networks/python
    # -convolutional-neural-network-creating-cnn-keras-tensorflow-plain-python/
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
    model.add(Dropout(0.2))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(layers.Dense(len(all_char), activation='softmax'))
    # model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    if train:
        model.fit(x_train, y_train,
                  batch_size=64,
                  epochs=10)

        model.save('data_char.keras')
        print('CNN model has been saved to the file data_char.keras')
    model2 = load_model('data_char.keras')
    score = model2.evaluate(x_test, y_test, verbose=8)
    print('Test accuracy of CNN:' + str(score[1]))


def construct_svc():
    """
    construct a Support Vector Classification model and save it to the file svc.sav
    """
    dataset_paths = glob.glob("dataset_characters/**/*.jpg")
    x = []
    labels = []
    size = (30, 60)
    for image_path in dataset_paths[::]:
        label = image_path.split(os.path.sep)[-2]
        img_num = image_path.split(os.path.sep)[-1]
        image_path = os.path.join('./dataset_characters', str(label), str(img_num))
        img1 = cv2.imread(image_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = cv2.resize(img1, size)
        # convert image into binary
        binary_image = img1 < threshold_otsu(img1)
        # flatten the image
        flattened_image = binary_image.reshape(-1)
        x.append(flattened_image)
        labels.append(label)
    x = np.asarray(x)
    labels = np.array(labels)
    x_train, x_test, y_train1, y_test1 = train_test_split(x, labels, test_size=0.25)
    svc_model = svm.LinearSVC()
    # fit the SVC model
    svc_model.fit(x_train, y_train1)
    # compute the test accuracy
    print("SVC model achieves accuracy:" + str(svc_model.score(x_test, y_test1)) + "on the test set")
    # save SVC model
    filename = './svc.sav'
    pickle.dump(svc_model, open(filename, 'wb'))
    print("SVC model is saved")


if __name__ == '__main__':
    # construct_dt()
    # construct_knn()
    # construct_svc()
    construct_cnn(train=False)
