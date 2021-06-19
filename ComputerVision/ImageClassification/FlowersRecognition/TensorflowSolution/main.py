import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class Variables:
  data_dir = "../input/flowers-recognition/flowers/flowers"
  batch_size = 32
  img_height = 180
  img_width = 180
  num_classes = 5
  epochs = 50


class FlowersRecognition():
  def dataloader_maker(self, data_dir, img_height, img_width, batch_size, ):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=41,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds

  def model_maker(self, img_height, img_width, num_classes):
      data_augmentation = keras.Sequential(
          [layers.experimental.preprocessing.RandomFlip("horizontal",
                                                        input_shape=(img_height,
                                                                     img_width,
                                                                     3)),
           layers.experimental.preprocessing.RandomRotation(0.1),
           layers.experimental.preprocessing.RandomZoom(0.1),
           ])
      model = Sequential([
          data_augmentation,
          layers.experimental.preprocessing.Rescaling(1. / 255),
          layers.Conv2D(16, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(32, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(64, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Dropout(0.2),
          layers.Flatten(),
          layers.Dense(128, activation='relu'),
          layers.Dense(num_classes)
      ])
      return model

  def train_model(self, model, train_ds, val_ds, epochs):
      model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

      reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                       patience=4, min_lr=0.0001)
      early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)

      history = model.fit(
          train_ds,
          validation_data=val_ds,
          epochs=epochs,
          callbacks=[reduce_lr, early_stop]
      )

      acc = history.history['accuracy']
      val_acc = history.history['val_accuracy']

      loss = history.history['loss']
      val_loss = history.history['val_loss']
      epochs_range = range(epochs)
      return acc, val_acc, loss, val_loss, epochs_range

  def show_results_training(self, acc, val_acc, loss, val_loss, epochs_range):
      plt.figure(figsize=(8, 8))
      plt.subplot(1, 2, 1)
      plt.plot(epochs_range, acc, label='Training Accuracy')
      plt.plot(epochs_range, val_acc, label='Validation Accuracy')
      plt.legend(loc='lower right')
      plt.title('Training and Validation Accuracy')

      plt.subplot(1, 2, 2)
      plt.plot(epochs_range, loss, label='Training Loss')
      plt.plot(epochs_range, val_loss, label='Validation Loss')
      plt.legend(loc='upper right')
      plt.title('Training and Validation Loss')
      plt.show()


if __name__ == '__main__':
  flowers = FlowersRecognition()
  train_ds, val_ds = flowers.dataloader_maker(Variables.data_dir, Variables.img_height,
                                              Variables.img_width, Variables.batch_size, )
  model = flowers.model_maker(Variables.img_height, Variables.img_width, Variables.num_classes)
  acc, val_acc, loss, val_loss, epochs_range = flowers.train_model(model, train_ds, val_ds, Variables.epochs)
  flowers.show_results_training(acc, val_acc, loss, val_loss, epochs_range)
