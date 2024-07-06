import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader
from torch.optim import Adam

import datetime
import argparse
import os
import logging

def parse_arguments():
    parser = argparse.ArgumentParser(description='Face Classification using EfficientNet')
    parser.add_argument('--job-name', type=str, required=True, help='Job name for identifying')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the root data directory')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes in the dataset')
    return parser.parse_args()

def load_data(data_dir, batch_size, val_split=0.2):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
    }

    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])

    num_train = int((1 - val_split) * len(full_dataset))
    num_val = len(full_dataset) - num_train

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [num_train, num_val])

    val_dataset.dataset.transform = data_transforms['val']

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }

    return dataloaders


def setup_model(num_classes):
    model = models.efficientnet_b7(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, num_classes),
        nn.Softmax(dim=1)
    )
    return model
    
def validate_model(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    val_loss = running_loss / len(dataloader.dataset)
    val_acc = running_corrects.double() / len(dataloader.dataset)

    return val_loss, val_acc

def train_model(model, criterion, optimizer, dataloaders, args, logger, device):
    num_epochs = args.num_epochs
    best_acc = 0.0

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        logger.info('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)

        logger.info(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation phase
        val_loss, val_acc = validate_model(model, criterion, dataloaders['val'], device)
        logger.info(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # Save the best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.best_model_path)
            logger.info(f'Best model saved at {args.best_model_path}')

    logger.info(f'Best validation accuracy: {best_acc:.4f}')
    logger.info(f'Best model saved at: {args.best_model_path}')

def main():
    args = parse_arguments()

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-')
    save_dir = os.path.join("runs", timestamp + args.job_name)
    os.makedirs(save_dir, exist_ok=True)
    
    best_model_path = os.path.join(save_dir, f'best_model.pth')
    args.best_model_path = best_model_path

    log_file_path = os.path.join(save_dir, 'training_log.txt')
    logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    dataloaders = load_data(args.data_dir, args.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = setup_model(args.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    model = model.to(device)

    train_model(model, criterion, optimizer, dataloaders, args, logger, device)

if __name__ == '__main__':
    main()
