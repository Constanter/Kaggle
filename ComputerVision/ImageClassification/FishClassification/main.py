from Variables import *
from GetData import Data
from Train import Train
from sklearn.model_selection import train_test_split


def main():
	data = Data()
	images_paths, labels_int, classes = data.get_path(root_dir)
	data, test_data, labels, test_labels = train_test_split(images_paths, labels_int, test_size=0.1, shuffle=True)
	train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.1, shuffle=True)
	train_transform, val_transform = data.get_transforms()
	train_loader, val_loader, test_loader = data.get_loaders(
													train_data,
													val_data, 
													train_labels, 
													val_labels,
													test_data,
													test_labels, 
													BATCH_SIZE, 
													train_transform, 
													val_transform)
													
	labels_dic = {i: label for i, label in enumerate(classes)}
	data.show_batch(train_loader, BATCH_SIZE, labels_dic)
	train = Train()
	model = train.fit(train_loader, val_loader, DEVICE, LEARNING_RATE)
	train.eval_accuracy(val_loader, model, DEVICE)


if __name__ == 'main':
	main()