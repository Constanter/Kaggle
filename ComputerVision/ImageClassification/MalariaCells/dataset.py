import tensorflow_datasets as tfds
import tensorflow as tf


TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
IM_SIZE = 224
BATCH_SIZE = 50


def resize(image, label):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE)) / 255, label


def create_dataset():
    dataset, info = tfds.load('malaria', with_info=True, as_supervised=True, shuffle_files=True, split='train')
    DATASET_SIZE = len(dataset)

    train_dataset = dataset.take(int(TRAIN_RATIO*DATASET_SIZE))
    val_dataset_test = dataset.skip(int(TRAIN_RATIO*DATASET_SIZE))
    val_dataset = val_dataset_test.take(int(VAL_RATIO*DATASET_SIZE))
    test_dataset = val_dataset_test.skip(int(TEST_RATIO*DATASET_SIZE))

    train_dataset = train_dataset.map(resize)
    val_dataset = val_dataset.map(resize)
    test_dataset = test_dataset.map(resize)

    train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(
        BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
