import torch

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 15
NUM_WORKERS = 2
IMAGE_HEIGHT = 160 
IMAGE_WIDTH = 240  
PIN_MEMORY = True
LOAD_MODEL = False
root_dir = '../input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset'
