import torch
import torchvision
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torchvision.models as models
from torchvision.utils import make_grid
import time
import copy

class Variables:
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 32
    # percentage of training set to use as validation
    valid_size = 0.2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 20
    data_dir = '../input/flowers-recognition/flowers/flowers'

class FlowerRecognition:
    def make_dataloader(self, data_dir, valid_size, batch_size, num_workers):
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        test_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

        dataset = datasets.ImageFolder(data_dir, transform=train_transforms)

        len_train_set = int(0.8*len(dataset))
        len_test_set = len(dataset) - len_train_set
        # repare datasets.train_data will be use for training,and test_data for final test
        train_data, test_data = torch.utils.data.random_split(dataset, [len_train_set, len_test_set])
        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # prepare data loaders (combine dataset and sampler)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,sampler=valid_sampler)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
        return train_loader, valid_loader, test_loader

    def make_model(self, device, ):
        # we are using  pretrained on ImageNet model resnet34
        model_conv = torchvision.models.resnet34(pretrained=True)
        for param in model_conv.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = torch.nn.Linear(num_ftrs, 5)

        model_conv = model_conv.to(device)

        optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

        criterion = torch.nn.CrossEntropyLoss()
        return model_conv, optimizer_conv, exp_lr_scheduler, criterion

    def train_model(self, model, criterion, optimizer, scheduler, device, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model


    def test_model(self, model, device):
        result = 0
        counter = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model_conv(inputs)
                _, preds = torch.max(outputs, 1)
                result += int(sum(labels == preds))
                counter += len(labels)
        print('Correct_answers - {0}, Total_answers - {1}, Percent_corrects - {2}'.format(result, counter, result / counter))

if __name__ == '__main__':
    flowers = FlowerRecognition()
    train_loader, valid_loader, test_loader = flowers.make_dataloader(Variables.data_dir, Variables.valid_size,
                                                                      Variables.batch_size, Variables.num_workers)
    dataloaders = {'train': train_loader, 'val': valid_loader}
    dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val']}
    model_conv, optimizer_conv, exp_lr_scheduler, criterion = flowers.make_model(Variables.device)
    model_ft = flowers.train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, Variables.device, Variables.num_epochs)
    flowers.test_model(model_ft, Variables.device)

