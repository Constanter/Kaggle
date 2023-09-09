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
