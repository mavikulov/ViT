from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index])
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.labels[index]).long()
        return image, label 
    

def get_cifar_loaders(
    X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32, shuffle_train=True      
):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]

    train_transform = T.Compose([
        T.RandomResizedCrop((32, 32), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        T.RandomRotation([-5, 5]),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.2),
        T.Normalize(mean, std)
    ])

    val_transform = T.Compose([
        T.Normalize(mean, std)
    ])

    test_transform = val_transform

    train_dataset = ImageDataset(X_train, y_train, train_transform)
    val_dataset = ImageDataset(X_val, y_val, val_transform)
    test_dataset = ImageDataset(X_test, y_test, test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
