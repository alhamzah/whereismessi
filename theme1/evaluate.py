import random
from random import randint
import os
from PIL import Image
import pprint

import numpy as np
import pandas as pd


DATASET_PREFIX = os.path.expanduser('~/data/theme1/dataset/')
IMAGES_PREFIX = os.path.join(DATASET_PREFIX, 'images')
random.seed(42)
IMG_SHAPE = (1920, 1080)
NUM_CLASSES = 11


def evaluate():
    img_paths = pd.read_csv(os.path.join(DATASET_PREFIX, 'test.csv'))['image_path'].to_list()
    print(f'Found {len(img_paths)} images')

    actuals, predictions = [], []
    for img_path in img_paths:
        img_path = os.path.join(IMAGES_PREFIX, img_path)
        img = Image.open(img_path)
        assert img.size == IMG_SHAPE

        predictions.append({
            c: [generate_random_prediction(c) for _ in range(randint(0, 5))]
            for c in range(NUM_CLASSES)
        })
        actuals.append({
            c: [generate_random_prediction(c) for _ in range(randint(0, 5))]
            for c in range(NUM_CLASSES)
        })
    assert len(actuals) == len(predictions)

    ious = np.zeros(len(actuals), NUM_CLASSES)
    for i, (actual, pred) in enumerate(zip(actuals, predictions)):
        for c in range(NUM_CLASSES):
            ious[i, c] = iou(actual[c], pred[c])

    thresholds = range(0.5, 1, 0.05)
    aps = np.zeros(len(thresholds, NUM_CLASSES)
    for thresh in thresholds:
        # TODO evaluate per class
        # waiting on clearer definition of which thresholds to use
        pass


def generate_random_prediction(c=None):
    c = c if c is not None else randint(0, NUM_CLASSES-1),
    xmin, ymin = randint(0, IMG_SHAPE[0]), randint(0, IMG_SHAPE[1])
    xmax, ymax = randint(xmin, IMG_SHAPE[0]), randint(ymin, IMG_SHAPE[1])
    return c, Rectangle(xmin, ymin, xmax, ymax)


class Rectangle:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    @property
    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)


def iou(actuals, preds):
    actual, pred = np.zeros(IMG_SHAPE), np.zeros(IMG_SHAPE)

    for r in actuals:
        actual[r.xmin:r.xmax, r.ymin:r.ymax] = 1
    for r in preds:
        pred[r.xmin:r.xmax, r.ymin:r.ymax] = 2

    combined = actual + pred
    fn = (combined == 1)
    fp = (combined == 2)
    tp = (combined == 3)
    return tp / (fn + fp + tp)


if __name__ == '__main__':
    evaluate()
