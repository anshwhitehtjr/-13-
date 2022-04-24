import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import cv2
from sklearn.metrics import accuracy_score
import sys

X = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]

classes = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

nclasses = len(classes)
print(len(pd.Series(y).value_counts()))

x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=9, train_size=3500, test_size=500
)

x_trained_scaled = x_train / 255
x_tested_scaled = x_test / 255
classifier = LogisticRegression(solver="saga", multi_class="multinomial").fit(
    x_trained_scaled, y_train
)


def get_prediction(img):
    img_pixels = Image.open(img)
    img_bw = img_pixels.convert("L")
    img_bw_resized = img_bw.resize((22, 30), Image.ANTIALIAS)
    PIXEL_FILTER = 20
    min_pixel = np.percentile(img_bw_resized, PIXEL_FILTER)
    img_bw_resized_inverted_scaled = np.clip(img_bw_resized - min_pixel, 0, 255)
    max_pixel = np.max(img_bw_resized)
    img_bw_resized_inverted_scaled = (
        np.asarray(img_bw_resized_inverted_scaled) / max_pixel
    )
    test_sample = np.array(img_bw_resized_inverted_scaled).reshape(1, 660)
    test_prediction = classifier.predict(test_sample)
    return test_prediction[0]
