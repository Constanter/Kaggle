import pandas as pd
import tensorflow as tf
import numpy as np

TRAIN_RATION = 0.8
TEST_RATION = 0.1


def create_dataset(path_to_data: str) -> np.array:
    # load data
    df = pd.read_csv(path_to_data)

    data = tf.constant(df, dtype=tf.float32)
    data = tf.random.shuffle(data)

    X = data[:, 3:-1]
    y = tf.expand_dims(data[:, -1], axis=-1)

    DATASET_SIZE = X.shape[0]

    X_train = X[:int(DATASET_SIZE * TRAIN_RATION)]
    y_train = y[:int(DATASET_SIZE * TRAIN_RATION)]

    X_test = X[int(DATASET_SIZE * TRAIN_RATION):int(DATASET_SIZE*(TEST_RATION + TRAIN_RATION))]
    y_test = y[int(DATASET_SIZE * TRAIN_RATION):int(DATASET_SIZE*(TEST_RATION + TRAIN_RATION))]

    X_val = X[int(DATASET_SIZE * (TEST_RATION + TRAIN_RATION)):]
    y_val = y[int(DATASET_SIZE * (TEST_RATION + TRAIN_RATION)):]

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, X_test, y_test
