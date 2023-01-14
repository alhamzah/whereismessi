import os
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


EfficientNetB0_INPUT_SIZE = (224, 224)
DATA_PATH = os.path.expanduser('~/data/mnist/MNIST Dataset JPG format/MNIST - JPG - training/')
SEED = 42


class CFG:
    dropout = 0
    batch_size = 16
    loss = 'huber'
    metrics = [keras.metrics.RootMeanSquaredError(), 'mean_absolute_error']
    epochs = 2
    workers = 2


def main():
    ds = get_image_ds()
    model, backbone = construct_model()
    train(model, backbone, ds)


def get_image_ds() -> tf.data.Dataset:
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_PATH,
        labels="inferred",
        batch_size=CFG.batch_size,
        image_size=EfficientNetB0_INPUT_SIZE,
        shuffle=True,
        seed=SEED,
    )
    counts = Counter(v for v in ds.map(lambda x, y: y).unbatch().as_numpy_iterator())
    print(f'Found labels {counts=}')
    return ds.cache().prefetch(tf.data.AUTOTUNE)


def train(model, backbone, ds):
    model.compile(
        loss=CFG.loss,
        optimizer=keras.optimizers.Adam(0.001),
        metrics=CFG.metrics
    )
    model.fit(
        ds,
        epochs=CFG.epochs,
        workers=CFG.workers
    )


def construct_model():
    model_input = keras.Input(
        shape=(*EfficientNetB0_INPUT_SIZE, 3),
        name='input_layer', dtype=tf.float32
    )
    x = model_input
    backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=x)
    backbone.trainable = False

    x = layers.GlobalAveragePooling2D(name='avg_pool')(backbone.output)
    x = layers.BatchNormalization()(x)
    if CFG.dropout:
        x = layers.Dropout(CFG.dropout, name='top_dropout')(x)
    outputs = layers.Dense(1, name='pred')(x)
    model = keras.Model(inputs=model_input, outputs=outputs, name='mlp')
    return model, backbone



if __name__ == '__main__':
    main()
