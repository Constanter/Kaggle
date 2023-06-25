import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Normalization, Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError, Huber, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt


def create_model(num_layers=5, depth_layers=32, activation='relu'):
    normalizer = Normalization()
    layers = [Dense(depth_layers, activation=activation) for _ in range(num_layers)]
    model = tf.keras.Sequential([
        InputLayer(input_shape=8),
        normalizer,
        *layers,
        Dense(1)
    ])
    return model


def get_visualizitons_training(history, need_to_show=False, loss=True):
    if loss:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val_loss'])
        if need_to_show:
            plt.show()
        else:
            plt.savefig('./results/loss.png')
    else:
        plt.plot(history.history['root_mean_squared_error'])
        plt.plot(history.history['val_root_mean_squared_error'])
        plt.title('model performance')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'])
        if need_to_show:
            plt.show()
        else:
            plt.savefig('./results/performance.png')


def train_model(model, train_dataset, val_dataset, X_test, y_test, lr=1e-3, epoch=100):

    # model.summary()
    model.compile(loss=MeanAbsoluteError(), optimizer=Adam(learning_rate=lr), metrics=RootMeanSquaredError())

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epoch, verbose=1)

    model.evaluate(X_test, y_test)

    return history, model


def get_visualizitons_test(X_test, y_test, model,  need_to_show=False):
    y_true = list(y_test[:, 0].numpy())
    y_pred = list(model.predict(X_test)[:, 0])
    width = 0.4
    ind = np.arange(100)
    plt.figure(figsize=(40, 12))
    plt.bar(ind, y_pred, width, label='Predicted Car Price')
    plt.bar(ind + width, y_true, width, label='Actual Car Price')

    plt.legend(['Predict', 'Actual'])
    plt.xlabel('Actual vs Predicted Prices')
    plt.ylabel('Car Price Prices')
    if need_to_show:
        plt.show()
    else:
        plt.savefig('./results/test_data_prediction.png')
