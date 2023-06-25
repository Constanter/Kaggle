from data_preparation import create_dataset
from training import (create_model,
                      get_visualizitons_training,
                      get_visualizitons_test,
                      train_model
                      )

LR = 0.001
EPOCH = 400

def main():
    path = './data/train.csv'
    train_dataset, val_dataset, X_test, y_test = create_dataset(path)
    model = create_model()
    history, model = train_model(model, train_dataset, val_dataset, X_test, y_test, lr=LR, epoch=EPOCH)
    get_visualizitons_training(history)
    get_visualizitons_training(history,loss=False)
    get_visualizitons_test(X_test, y_test, model)


if __name__ == '__main__':
    main()
