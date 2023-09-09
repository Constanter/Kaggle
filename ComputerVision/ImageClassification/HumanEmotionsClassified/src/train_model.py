import hydra
from omegaconf import DictConfig
from simple_cnn import simple_cnn
from get_dataset import get_datasets
from visualization import get_visualizitons_training

CLASS_NAMES = ('angry', 'happy', 'sad')


@hydra.main(config_path="../config", config_name="main", version_base=None)
def train_model(config: DictConfig):
    """Function to train the model"""
    print(f"Model used: {config.model.name}")
    print(f"Save the output to {config.data.result}")

    epochs = config.train_param.epochs
    batch_size = config.train_param.batch_size
    im_size = config.train_param.im_size
    model_params = config.model
    dataset_folder = config.data.raw
    save_path = config.data.result

    training_dataset, validation_dataset = get_datasets(dataset_folder, batch_size, im_size, CLASS_NAMES)
    model = simple_cnn(**model_params)

    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        verbose=1
    )
    get_visualizitons_training(history, save_path)
    get_visualizitons_training(history, save_path, loss=False)


if __name__ == "__main__":
    train_model()
