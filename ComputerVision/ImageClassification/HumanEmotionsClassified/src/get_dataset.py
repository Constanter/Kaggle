import tensorflow as tf


def get_datasets(data_folder: str, batch_size: int, im_size: int, class_names: tuple) -> tuple:
    train_directory = f'{data_folder}/train'
    test_directory = f'{data_folder}/test'

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_directory,
        labels='inferred',
        label_mode='categorical',
        class_names=class_names,
        color_mode='rgb',
        batch_size=im_size,
        image_size=(im_size, im_size),
        shuffle=True,
        seed=13
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        test_directory,
        labels='inferred',
        label_mode='categorical',
        class_names=class_names,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(im_size, im_size),
        shuffle=False,
        seed=13
    )

    training_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    return training_dataset, validation_dataset
