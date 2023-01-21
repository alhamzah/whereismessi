import argparse
import os
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

TEMPE_CSV_PATH = os.path.expanduser('~/projects/smartathon_2023/datasets/theme2/tempe_data_processed.csv')
TEMPE_IMG_PATH = os.path.expanduser('~/projects/smartathon_2023/datasets/theme2/tempe_images/')

MODEL_CHECKPOINT_DIR = os.path.expanduser('~/projects/smartathon_2023/theme2/model_checkpoints/')

EfficientNetB0_INPUT_SIZE = (224, 224)
EfficientNetB3_INPUT_SIZE = (300, 300)
EfficientNetB4_INPUT_SIZE = (380, 380)
EfficientNetB5_INPUT_SIZE = (456, 456)

INPUT_SIZE = EfficientNetB0_INPUT_SIZE
BACKBONE = tf.keras.applications.EfficientNetV2B0

SEED = 42
SHUFFLE_BUFFER_SIZE = 10**6


class CFG:
    dropout = 0.1
    training_batch_size = 32
    finetuning_batch_size = 32
    loss = 'huber'
    metrics = [keras.metrics.RootMeanSquaredError(), 'mean_absolute_error']
    epochs = 50
    workers = 8


def main(model_name):
    ds = get_image_ds()
    model, backbone = construct_model()
    train(model_name, model, backbone, ds)


def image_label_generator():
    tempe_data = pd.read_csv(TEMPE_CSV_PATH)
    for filename in os.listdir(TEMPE_IMG_PATH):
        section_num = filename.split('_')[0]
        label = tempe_data.loc[tempe_data.SECTION_NUMBER == int(section_num)].CURRENT_PQI.iloc[0]
        path = os.path.join(TEMPE_IMG_PATH, filename)
        yield (path, label)


def prepare_image(img):
    return tf.image.resize(
        tf.io.decode_image(
            img, channels=3, expand_animations=False, dtype=tf.uint8
        ),
        INPUT_SIZE
    )


def get_image_ds() -> tf.data.Dataset:
    paths_ds = tf.data.Dataset.from_generator(image_label_generator, output_types=(tf.string, tf.float32))
    image_ds = paths_ds.map(
        lambda p, ll: (prepare_image(tf.io.read_file(p)), ll),
        num_parallel_calls=tf.data.AUTOTUNE)
    return image_ds.shuffle(SHUFFLE_BUFFER_SIZE).cache().prefetch(tf.data.AUTOTUNE)


def train(model_name, model, backbone, ds):
    ds_size = len([_ for _ in ds])
    train_size = int(ds_size * 0.8)
    print(f'Training using {train_size=}')
    train_ds, val_ds = ds.take(train_size), ds.skip(train_size)
    monitored_metric = 'val_mean_absolute_error'

    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor=monitored_metric,
        factor=0.5,
        patience=3,
        min_lr=1e-4, verbose=1
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor=monitored_metric, patience=20, restore_best_weights=True, verbose=2
    )
    checkpoint_path = os.path.join(MODEL_CHECKPOINT_DIR, model_name)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor=monitored_metric, verbose=2, save_best_only=True, save_weights_only=False
    )
    training_cbs = [reduce_lr_cb, early_stopping_cb, checkpoint_cb]
    training_cbs = [reduce_lr_cb, early_stopping_cb]

    for layer in backbone.layers:
        if layer.name == 'block6d_se_excite':
            set_trainable = True

    print('Compiling...')
    model.compile(
        loss=CFG.loss,
        optimizer=keras.optimizers.Adam(1e-3),
        metrics=CFG.metrics,
    )
    print('Training...')
    model.fit(
        train_ds.batch(CFG.training_batch_size),
        validation_data=val_ds.batch(CFG.training_batch_size),
        epochs=CFG.epochs,
        workers=CFG.workers,
        callbacks=training_cbs
    )
    print('Finished training')

    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor=monitored_metric,
        factor=0.5,
        patience=5,
        min_lr=1e-6, verbose=1
    )
    finetuning_cbs = [reduce_lr_cb, early_stopping_cb, checkpoint_cb]
    finetuning_cbs = [reduce_lr_cb, early_stopping_cb]

    backbone.trainable = True
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        loss=CFG.loss,
        optimizer=keras.optimizers.Adam(5e-4),
        metrics=CFG.metrics
    )
    print('Finetuning...')
    model.fit(
        train_ds.batch(CFG.finetuning_batch_size),
        validation_data=val_ds.batch(CFG.finetuning_batch_size),
        epochs=CFG.epochs,
        workers=CFG.workers,
        callbacks=finetuning_cbs
    )
    print(f'Saving model to {checkpoint_path}')
    model.save(checkpoint_path)


def construct_model():
    model_input = keras.Input(
        shape=(*INPUT_SIZE, 3),
        name='input_layer', dtype=tf.float32
    )
    x = model_input
    backbone = BACKBONE(include_top=False, weights='imagenet', input_tensor=x)
    backbone.trainable = False

    x = layers.GlobalAveragePooling2D(name='avg_pool')(backbone.output)
    x = layers.BatchNormalization()(x)
    if CFG.dropout:
        x = layers.Dropout(CFG.dropout, name='top_dropout')(x)
    outputs = layers.Dense(1, name='pred')(x)
    model = keras.Model(inputs=model_input, outputs=outputs, name='mlp')
    return model, backbone


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True, type=str)

    opt = parser.parse_args()
    print(opt)
    assert not os.path.isdir(os.path.join(MODEL_CHECKPOINT_DIR, opt.model_name))

    main(opt.model_name)
