import cv2
import tensorflow as tf
from github.Kaggle.ComputerVision.ImageClassification.HumanEmotionsClassified.src.model import create_model
from get_dataset import get_datasets


EPOCHS = 3
CLASS_NAMES = ('angry', 'happy', 'sad')


folder = '/home/tekhin/Downloads/Emotions Dataset/'
training_dataset, validation_dataset = get_datasets(folder)
model = create_model()
# print(model.summary())
#
history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    verbose=1
)

test_image = cv2.imread('/home/tekhin/Downloads/sad2.jpeg')
im = tf.constant(test_image, dtype=tf.float32)
im = tf.expand_dims(im, axis=0)
print(model(im))
