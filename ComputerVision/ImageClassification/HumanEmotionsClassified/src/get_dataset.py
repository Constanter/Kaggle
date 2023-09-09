import tensorflow as tf
from tensorflow.keras.layers import Resizing, Rescaling

CLASS_NAMES = ('angry', 'happy', 'sad')
CONFIGURATION = {'BATCH_SIZE': 16,
                 'IM_SIZE': 224}


def get_datasets(data_folder):
    train_directory = f'{data_folder}/train'
    test_directory = f'{data_folder}/test'

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_directory,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=CONFIGURATION['BATCH_SIZE'],
        image_size=(CONFIGURATION['IM_SIZE'], CONFIGURATION['IM_SIZE']),
        shuffle=True,
        seed=13
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        test_directory,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=CONFIGURATION['BATCH_SIZE'],
        image_size=(CONFIGURATION['IM_SIZE'], CONFIGURATION['IM_SIZE']),
        shuffle=True,
        seed=13
    )

    training_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    return training_dataset, validation_dataset
