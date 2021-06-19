import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold


class DigitRecognizer:
    def prepare_data(self,):
        #Load data from csv files
        data_train = pd.read_csv("../input/digit-recognizer/train.csv")
        X_test = pd.read_csv("../input/digit-recognizer/test.csv")

        # Split data and labels and normalize images
        X, y = data_train.drop(labels = ["label"],axis = 1)/255.,data_train["label"]
        X_test = X_test/255.

        # Reshape images to (batch_size x height x width x channels)
        X = X.values.reshape(-1,28,28,1)
        X_test = X_test.values.reshape(-1,28,28,1)
        return X, X_test, y

    def train_model(self, X, X_test, y):
        #Define a training strategy we reduct learning after 3 epochs
        # without improvements,and stop the train after 30 epochs without improvements
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=3,# How many epochs to wait before reduct lerning rate
                                                    verbose=1,
                                                    factor=0.3,
                                                    min_lr=0.00001)
        early_stopping = EarlyStopping(
            min_delta=0.000001, # minimium amount of change to count as an improvement
            patience=20, # how many epochs to wait before stopping
            restore_best_weights=True,
        )


        skf = StratifiedKFold(n_splits=3,random_state=42,shuffle=True)
        sub = pd.DataFrame(data=None, index=(range(1,28001)), columns=None, dtype=None, copy=False)
        for train_index, val_index in skf.split(X, y):
            model = keras.Sequential([
            keras.layers.Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',
                         activation ='relu', input_shape = (28,28,1)),
            keras.layers.MaxPool2D(pool_size=(2,2)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                         activation ='relu'),
            keras.layers.MaxPool2D(pool_size=(2,2)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            keras.layers.Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',
                         activation ='relu'),
            keras.layers.MaxPool2D(pool_size=(2,2)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            keras.layers.Flatten(),
            keras.layers.Dense(28*28, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(28*28, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(28*28, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
            ])
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
            history = model.fit(
            x=X_train, y=y_train, batch_size=100, epochs=250, verbose=1, callbacks=[early_stopping,learning_rate_reduction],
            validation_data=(X_val,y_val), shuffle=True)
            # predict results
            results = model.predict(X_test)
            # select the index with the maximum probability
            results = np.argmax(results,axis = 1)
            results = pd.Series(results,name="Label")
            sub = pd.concat([sub, results],axis=1)
            return sub

    def make_submission(self, sub):
        sub["result"] = sub.mode(dropna=True,axis=1)[0]
        result = pd.Series(sub["result"],name="Label")
        submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)
        submission = submission.dropna().astype('int32')
        submission.to_csv("mnist_ansamble_of_cnn.csv",index=False)


if __name__ == '__main__':
    digit_recognizer = DigitRecognizer()
    X, X_test, y = digit_recognizer.prepare_data()
    sub = digit_recognizer.train_model(X, X_test, y)
    digit_recognizer.make_submission(sub)
