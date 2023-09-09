import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns


def get_visualizitons_training(history, save_path, need_to_show=False, loss=True):
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
            plt.savefig(f'{save_path}/loss.png')
        plt.close()
    else:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model performance')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'])
        if need_to_show:
            plt.show()
        else:
            plt.savefig(f'{save_path}/performance.png')
        plt.close()


def plot_predictions(dataset, save_path, names, model):
    for image, label in dataset.take(1):
        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(image[i] / 255.)
            true_label = names[tf.argmax(label[i], axis=0).numpy()]
            predicted_label = names[tf.argmax(model(tf.expand_dims(image[i], axis=0)), axis=-1).numpy()[0]]
            plt.title(f'True_lbl: {true_label}, \n pred_lbl: {predicted_label}')
            plt.axis('off')
        plt.savefig(f'{save_path}/predictions_image.png')


def get_confusion_matrix(dataset, model, save_path):
    predicted = []
    labels = []
    for image, label in dataset:
        predicted.append(model(image))
        labels.append(label.numpy())
    predicted = np.argmax(predicted[:-1], axis=-1).flatten()
    labels = np.argmax(labels[:-1], axis=-1).flatten()

    cm = confusion_matrix(labels, predicted)
    plt.figure(figsize=(13, 13))
    sns.heatmap(cm, annot=True)

    plt.title('Confusion matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'{save_path}/confusion_matrix.png')


