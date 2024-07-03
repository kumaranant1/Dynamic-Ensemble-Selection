# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 16:56:23 2021

@author: ashra
"""
import torch
from torchvision import datasets, transforms
# import medmnist
# from medmnist import INFO
GENERATOR_SEED = 42


def create_data_transforms_CIFAR10(batch_size, train=True):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.CIFAR10(root='./data/cifar10', train=True,
                               transform=transform, download=True)
    data_gen = torch.Generator().manual_seed(GENERATOR_SEED)
    train_set, val_set = torch.utils.data.random_split(dataset, [40000, 10000],
                                                       generator=data_gen)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(
                    datasets.CIFAR10('./data/cifar10',download=True, train=False,
                                     transform=transform),
                    batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def create_data_transforms_CIFAR100(batch_size, train=True):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5)),
                                    ])
    dataset = datasets.CIFAR100(root='./data', train=True, transform=transform,
                                download=True)
    data_gen = torch.Generator().manual_seed(GENERATOR_SEED)
    train_set, val_set = torch.utils.data.random_split(dataset, [40000, 10000],
                                                       generator=data_gen)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(
                            datasets.CIFAR100('data/cifar100', train=False,
                                              download=True,
                                              transform=transform),
                            batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def create_data_transforms_MNIST(batch_size, train=True):
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x:
                                                      x.repeat(3, 1, 1)),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5)),
                                    ])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform,
                             download=True)
    data_gen = torch.Generator().manual_seed(GENERATOR_SEED)
    train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000],
                                                       generator=data_gen)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(
                            datasets.MNIST('data/mnist', train=False, download=True,
                                           transform=transform),
                            batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def create_data_transforms_EMNIST(batch_size, train=True):
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x:
                                                      x.repeat(3, 1, 1)),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5)),
                                    ])
    dataset = datasets.EMNIST(root='./data', train=True, split="balanced",
                              transform=transform,
                              download=True)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    data_gen = torch.Generator().manual_seed(GENERATOR_SEED)
    train_set, val_set = torch.utils.data.random_split(dataset,
                                                       [train_size, val_size],
                                                       generator=data_gen)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(
                            datasets.EMNIST('data/emnist', train=False,
                                            split="balanced",
                                            download=True,
                                            transform=transform),
                            batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def create_data_transforms_QMNIST(batch_size, train=True):
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x:
                                                      x.repeat(3, 1, 1)),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5)),
                                    ])
    dataset = datasets.QMNIST(root='./data', train=True,
                              transform=transform,
                              download=True)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    data_gen = torch.Generator().manual_seed(GENERATOR_SEED)
    train_set, val_set = torch.utils.data.random_split(dataset,
                                                       [train_size, val_size],
                                                       generator=data_gen)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(
                            datasets.QMNIST('data/qmnist', train=False,
                                            download=True,
                                            transform=transform),
                            batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def create_data_transforms_fashionMNIST(batch_size, train=True):
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x:
                                                      x.repeat(3, 1, 1)),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5)),
                                    ])
    dataset = datasets.FashionMNIST(root='./data', train=True,
                                    transform=transform,
                                    download=True)
    data_gen = torch.Generator().manual_seed(GENERATOR_SEED)
    train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000],
                                                       generator=data_gen)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(
                            datasets.FashionMNIST('data/fashionMNIST',
                                                  train=False, download=True,
                                                  transform=transform),
                            batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def create_data_transforms_SVHN(batch_size, train=True):
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5)),
                                    ])
    dataset = datasets.SVHN(root='./data', split="train", transform=transform,
                            download=True)
    data_gen = torch.Generator().manual_seed(GENERATOR_SEED)
    train_set, val_set = torch.utils.data.random_split(dataset, [58606, 14651],
                                                       generator=data_gen)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(
                        datasets.SVHN('data/SVHN', split="test", download=True,
                                      transform=transform),
                        batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# def create_data_transforms_MEDMNIST(data_flag, resize_x, resize_y,
#                                     download, batch_size,
#                                     train_shuffle=True, val_shuffle=False):

#     info = INFO[data_flag]
#     n_channels = info['n_channels']
#     n_classes = len(info['label'])
#     DataClass = getattr(medmnist, info['python_class'])

#     # preprocessing
#     data_transform = transforms.Compose([
#                         transforms.Resize((resize_x, resize_y)),
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean=[.5], std=[.5])
#                     ])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform,
                              download=download)
    val_dataset = DataClass(split='val', transform=data_transform,
                            download=download)
    test_dataset = DataClass(split='test', transform=data_transform,
                             download=download)
    # encapsulate data into dataloader form
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=train_shuffle)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=val_shuffle)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return n_channels, n_classes, train_loader, val_loader, test_loader



a,b = torch.utils.data.random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
for i in a:
    print(i)
print("**********")
for j in b:
    print(j)