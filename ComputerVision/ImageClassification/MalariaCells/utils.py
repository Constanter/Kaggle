import matplotlib.pyplot as plt


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
            plt.savefig('./results/performance.png')
        plt.close()


def get_prediction_type(probability):
    if probability > 0.5:
        return 'P'
    else:
        return 'U'


def save_predicted_images(dataset, model):
    for i, (img, lbl) in enumerate(dataset.take(16)):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(img[0])
        plt.title(str(get_prediction_type(lbl.numpy()[0])) + ':' + str(get_prediction_type(model.predict(img)[0][0])))
        plt.axis('off')
    plt.savefig('./results/prediction_example.png')

