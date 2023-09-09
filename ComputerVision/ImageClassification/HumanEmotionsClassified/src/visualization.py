import matplotlib.pyplot as plt
import tensorflow as tf


def visualize(dataset, names):
    plt.figure(figsize=(12, 12))
    for im, lbl in dataset.take(1):
        for i in range(16):
            ax = plt.subplot(4, 4, i+1)
            plt.imshow(im[i]/255.)
            plt.title(names[tf.argmax(lbl[i], axis=0).numpy()])
            plt.axis('off')
        plt.show()


def get_visualizitons_training(history, save_path,need_to_show=False, loss=True):
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