from tensorflow.keras.layers import InputLayer, Conv2D, Dense, MaxPool2D, Flatten, BatchNormalization, Rescaling, Resizing
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy,TopKCategoricalAccuracy
import tensorflow as tf


class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate: int):
        self.initial_learning_rate = initial_learning_rate
        self.counter = 1

    def __call__(self, step: int):
        rate = self.initial_learning_rate / self.counter
        self.counter += 1
        return rate

    def get_config(self):
        config = {
            'initial_learning_rate': self.initial_learning_rate,
            'counter': self.counter,
        }
        return config


def simple_cnn(
         num_classes: int,
         lr: int = 1e-3,
         im_size: int = 224,
         filters: int = 6,
         channels: int = 3,
         kernel_size: int = 3,
         pool_size: int = 2,
         strides: int = 2,
         activation: str = 'relu',
         dense_nodes: int = 100,
        **kwargs) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            Resizing(im_size, im_size),
            Rescaling(1. / 255),
            InputLayer(input_shape=(im_size, im_size, channels)),
            Conv2D(filters=filters, kernel_size=kernel_size, padding='valid', activation=activation),
            BatchNormalization(),
            MaxPool2D(pool_size=pool_size, strides=strides),

            Conv2D(filters=filters*3, kernel_size=kernel_size, padding='valid', activation=activation),
            BatchNormalization(),
            MaxPool2D(pool_size=pool_size, strides=strides),

            Flatten(),
            Dense(dense_nodes, activation=activation),
            BatchNormalization(),
            Dense(dense_nodes//10, activation=activation),
            BatchNormalization(),
            Dense(num_classes, activation='softmax'),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=MyLRSchedule(lr)),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy(name='accuraccy'), TopKCategoricalAccuracy(k=2, name='top_k_acc')]
    )
    return model
