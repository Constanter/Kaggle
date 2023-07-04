import os

from dataset import create_dataset
from model import create_model
from utils import save_predicted_images, get_visualizitons_training


def train(model, train_dataset, val_dataset):
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=10, verbose=1)

    return history


if __name__ == "__main__":
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda-12.2/"
    train_dataset, val_dataset, test_dataset = create_dataset()
    model = create_model()
    history = train(model, train_dataset, val_dataset)
    get_visualizitons_training(history, loss=False)
    get_visualizitons_training(history)
    # need to plot history train loss, train metrics
    model.evaluate(test_dataset)
    save_predicted_images(test_dataset, model)
    model.save('./results/malaria.hdf5')
