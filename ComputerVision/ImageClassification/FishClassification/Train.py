from torch import nn, optim
import torch
from catalyst import dl
from torchvision import models


class Train:
	def fit(self, train_loader, val_loader, device, lr):
		loaders = {
			"train": train_loader,
			"valid": val_loader,
		}

		mobilenet_v2 = models.mobilenet_v2()
		mobilenet_v2.classifier = nn.Sequential(
			nn.Dropout(p=0.2, inplace=False),
			nn.Linear(in_features=1280, out_features=9, bias=True))
		model = mobilenet_v2.to(device)

		optimizer = optim.Adam(model.parameters(), lr=lr)
		loss_fn = nn.CrossEntropyLoss()
		
		runner = dl.SupervisedRunner(
			input_key="features", output_key="logits", target_key="targets", loss_key="loss"
		)
		# model training
		runner.train(
			model=model,
			criterion=loss_fn,
			optimizer=optimizer,
			loaders=loaders,
			num_epochs=15,
			callbacks=[
				dl.AccuracyCallback(input_key="logits", target_key="targets", topk_args=(1, 3)),
				dl.EarlyStoppingCallback(
					patience=2, loader_key="valid", metric_key="loss", minimize=True),
				],
			logdir="./logs",
			valid_loader="valid",
			valid_metric="loss",
			minimize_valid_metric=True,
			verbose=True,
			load_best_on_end=True,
		)
		return model

	def eval_accuracy(self, loader, model, device):
		model.eval()
		corrects = 0
		total = 0
		for images, labels in loader:
			images, labels = images.to(device), labels.to(device)
			predictions = model(images)
			predict = torch.max(predictions.data, 1)[1].to(device)
			total += len(labels)
			corrects += (predict == labels).sum()
		accuracy = 100 * corrects / float(total)
		print(f' Accuracy: {accuracy}"')
