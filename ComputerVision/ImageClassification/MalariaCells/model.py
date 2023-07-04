from tensorflow.keras.layers import InputLayer, Conv2D, Dense, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


IM_SIZE = 224


class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate
        self.counter = 1

    def __call__(self, step):
        rate = self.initial_learning_rate / self.counter
        self.counter += 1
        return rate

    def get_config(self):
        config = {
            'initial_learning_rate': self.initial_learning_rate,
            'counter': self.counter,
        }
        return config


def create_model():
    model = tf.keras.Sequential(
        [
            InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
            Conv2D(filters=6, kernel_size=3, padding='valid', activation='relu'),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),

            Conv2D(filters=16, kernel_size=3, padding='valid', activation='relu'),
            BatchNormalization(),
            MaxPool2D(pool_size=2, strides=2),

            Flatten(),
            Dense(100, activation='relu'),
            BatchNormalization(),
            Dense(10, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid'),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=MyLRSchedule(1e-3)),
        loss=BinaryCrossentropy(),
        metrics='accuracy'
    )
    return model
