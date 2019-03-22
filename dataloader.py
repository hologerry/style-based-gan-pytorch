from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def lsun_loader(path, batch_size):
    def loader(transform):
        data = datasets.LSUNClass(
            path, transform=transform,
            target_transform=lambda x: 0)
        data_loader = DataLoader(data, shuffle=False, batch_size=batch_size, num_workers=8)

        return data_loader

    return loader


def celeba_loader(path, batch_size):
    def loader(transform):
        data = datasets.ImageFolder(path, transform=transform)
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size, num_workers=8)

        return data_loader

    return loader


def zi_loader(path, batch_size):
    def loader(transform):
        data = datasets.ImageFolder(path, transform=transform)
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size, num_workers=8)

        return data_loader

    return loader


def sample_data(dataloader, image_size=4):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    loader = dataloader(transform)

    return loader

    # for img, label in loader:
    #     yield img, label
